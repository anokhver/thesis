from django.conf import settings
from django.contrib import messages
from django.core.exceptions import ValidationError
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_POST

from .forms import CheckpointUploadForm, NewRunForm
from .models import AnalysisRun
from .services import jobs
from .services.checkpoints import (
    delete_checkpoint as delete_checkpoint_file,
    list_checkpoints,
    resolve_checkpoint,
    save_uploaded_checkpoint,
)
from .services.naming import GROUP_TREATMENTS, KNOWN_GROUPS
from .services.run_artifacts import ensure_run_dirs
from .services.artifacts import (
    categorize_plots,
    cluster_color_map,
    list_plots,
    output_dir_present,
)
from .services.run_uploads import (
    inspect_bundle,
    inspect_patches,
    safe_extract_zip,
)
from .services.training_config import (
    PREPROCESSING_DEFAULTS,
    PRETRAIN_CONFIG_PATH,
    grouped_pretrain_config,
)


def run_detail(request, run_id):
    run = get_object_or_404(AnalysisRun, id=run_id)
    source_stats = list(run.source_stats.all())

    plots = list_plots(run)
    color_map = cluster_color_map(source_stats)
    annotated_stats = []
    for s in source_stats:
        cid = s.dominant_cluster
        annotated_stats.append({
            "source_image": s.source_image,
            "treatment_group": s.treatment_group,
            "n_patches": s.n_patches,
            "dominant_cluster": cid,
            "dominant_cluster_color": (
                color_map.get(cid) if cid is not None else None
            ),
        })

    served = [p for p in plots if p["url"] is not None]
    unserved = [p for p in plots if p["url"] is None]
    plots_by_name = categorize_plots(plots)
    has_group_plots = any(
        plots_by_name[k] is not None and plots_by_name[k].get("url")
        for k in ("umap_groups", "group_heatmap", "group_boxplots")
    )

    return render(
        request,
        "synapse_web/run_detail.html",
        {
            "run": run,
            "source_stats": annotated_stats,
            "expected_count": run.expected_image_count,
            "thesis_pdf_url": settings.THESIS_PDF_URL,
            "plots": plots,
            "served_count": len(served),
            "unserved_count": len(unserved),
            "plots_by_name": plots_by_name,
            "has_group_plots": has_group_plots,
            "output_dir_present": output_dir_present(run),
            "debug": settings.DEBUG,
        },
    )


@never_cache
def run_status(request, run_id):
    run = get_object_or_404(AnalysisRun, id=run_id)
    return JsonResponse(
        {
            "status": run.status,
            "progress": run.progress,
            "progress_message": run.progress_message,
            "result_count": run.source_stats.count(),
            "expected_count": run.expected_image_count,
            "finished": run.status in ("completed", "failed"),
            "error_message": run.error_message,
        }
    )


def results(request):
    latest_run = AnalysisRun.objects.filter(status="completed").first()

    run_summaries = []
    for run in AnalysisRun.objects.all()[:20]:
        run_summaries.append(
            {
                "id": run.id,
                "id_short": str(run.id)[:8],
                "created_at": run.created_at,
                "status": run.status,
                "run_kind": run.get_run_kind_display(),
                "n_source_images": run.n_source_images,
                "n_patches": run.n_patches,
                "n_clusters": run.n_clusters,
                "expected_count": run.expected_image_count,
                "checkpoint_path": run.checkpoint_path or "—",
            }
        )

    return render(
        request,
        "synapse_web/results.html",
        {
            "run_summaries": run_summaries,
            "latest_run": latest_run,
        },
    )


def overview(request):
    config_sections = grouped_pretrain_config()
    treatments = [
        {"token": tok, "treatment": GROUP_TREATMENTS[tok]}
        for tok in KNOWN_GROUPS
    ]
    return render(
        request,
        "synapse_web/overview.html",
        {
            "config_sections": config_sections,
            "config_available": bool(config_sections),
            "config_filename": PRETRAIN_CONFIG_PATH.name,
            "preprocessing": PREPROCESSING_DEFAULTS,
            "treatments": treatments,
            "thesis_pdf_url": settings.THESIS_PDF_URL,
        },
    )


def checkpoints(request):
    if request.method == "POST":
        form = CheckpointUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                final_path = save_uploaded_checkpoint(form.cleaned_data["file"])
            except ValidationError as exc:
                for msg in exc.messages:
                    form.add_error("file", msg)
            else:
                messages.success(
                    request, f"Uploaded checkpoint {final_path.name!r}."
                )
                return redirect("synapse_web:checkpoints")
    else:
        form = CheckpointUploadForm()

    return render(
        request,
        "synapse_web/checkpoints.html",
        {
            "form": form,
            "checkpoints": list_checkpoints(),
            "max_upload_bytes": int(settings.MAX_CHECKPOINT_UPLOAD_BYTES),
        },
    )


@require_POST
def delete_checkpoint(request):
    name = (request.POST.get("name") or "").strip()
    if not name:
        messages.error(request, "Missing checkpoint name.")
    elif delete_checkpoint_file(name):
        messages.success(request, f"Deleted checkpoint {name!r}.")
    else:
        messages.error(request, f"Could not delete checkpoint {name!r}.")
    return redirect("synapse_web:checkpoints")


def _create_run_from_form(form: NewRunForm, archive) -> tuple[AnalysisRun, list[str]]:
    """Create the AnalysisRun, extract + inspect, populate manifest.

    Returns (run, unknown_source_images). Raises ValidationError on
    any inspection failure; the caller is responsible for deleting the
    half-built run row so the pre_delete signal cleans the dir.
    """
    run_kind = form.cleaned_data["run_kind"]
    ckpt_name = form.cleaned_data.get("checkpoint") or ""
    ckpt_path = ""
    if ckpt_name:
        resolved = resolve_checkpoint(ckpt_name)
        if resolved is None:
            raise ValidationError(
                "Selected checkpoint disappeared between page load and submit."
            )
        ckpt_path = str(resolved)

    with transaction.atomic():
        run = AnalysisRun.objects.create(
            status="pending",
            run_kind=run_kind,
            checkpoint_path=ckpt_path,
            commit_sha=jobs.current_commit_sha(),
            config_snapshot={
                "checkpoint_name": ckpt_name,
                "checkpoint_path": ckpt_path,
                "work_fn": settings.ANALYSIS_WORK_FN,
                "source": "new_run_form",
            },
        )
        input_dir, bundle_dir, output_dir = ensure_run_dirs(run)
        safe_extract_zip(archive, input_dir)

        if run_kind == "extract_and_cluster":
            manifest = inspect_patches(input_dir)
            persisted_bundle_dir = str(bundle_dir)
        else:
            manifest, bundle_root = inspect_bundle(input_dir)
            persisted_bundle_dir = str(bundle_root)

        run.input_manifest = manifest
        run.n_patches = sum(int(item["n_patches"]) for item in manifest)
        run.n_source_images = len(manifest)
        run.input_dir = str(input_dir)
        run.bundle_dir = persisted_bundle_dir
        run.output_dir = str(output_dir)
        run.save(update_fields=[
            "input_manifest",
            "n_patches",
            "n_source_images",
            "input_dir",
            "bundle_dir",
            "output_dir",
        ])

    unknown = [m["source_image"] for m in manifest if not m["treatment_group"]]
    return run, unknown


def new_run(request):
    if request.method == "POST":
        form = NewRunForm(request.POST, request.FILES)
        if form.is_valid():
            archive = form.cleaned_data["archive"]
            run: AnalysisRun | None = None
            try:
                run, unknown = _create_run_from_form(form, archive)
            except ValidationError as exc:
                if run is not None:
                    run.delete()
                for msg in exc.messages:
                    form.add_error(None, msg)
            else:
                if unknown:
                    messages.warning(
                        request,
                        f"No treatment-group token recognised in "
                        f"{len(unknown)} source image(s); they will appear "
                        "as 'UNKNOWN' in summaries.",
                    )
                jobs.submit_run(str(run.id))
                messages.success(request, "Run created and queued.")
                return redirect("synapse_web:run_detail", run_id=run.id)
    else:
        form = NewRunForm()

    return render(
        request,
        "synapse_web/new_run.html",
        {
            "form": form,
            "checkpoints": list_checkpoints(),
            "max_zip_bytes": int(settings.MAX_ZIP_UPLOAD_BYTES),
        },
    )
