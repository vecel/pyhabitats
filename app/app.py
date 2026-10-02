import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import tempfile

    import marimo as mo
    import numpy as np
    from PIL import Image

    from pyhabitats import HabitatsExtractor
    from pyhabitats.data_loader import DataLoader

    return DataLoader, HabitatsExtractor, Image, mo, np, os, tempfile


@app.cell
def _(mo):
    image_upload = mo.ui.file(label="Upload Image", kind="area")
    mask_upload = mo.ui.file(label="Upload Mask", kind="area")
    return image_upload, mask_upload


@app.cell
def _(image_upload, mask_upload, mo):
    mo.vstack(
        [
            mo.md("# Habitat Analysis"),
            mo.md("Upload image and its mask."),
            image_upload,
            mask_upload,
        ]
    )


@app.cell
def _(handle_upload, image_upload, mask_upload):
    image_path, image_array, image_view = handle_upload(image_upload)
    mask_path, mask_array, mask_view = handle_upload(mask_upload)
    return image_array, image_path, image_view, mask_path, mask_view


@app.cell
def _(render, save_to_tmp, to_array):
    def handle_upload(uploaded_file):
        path = save_to_tmp(uploaded_file)
        array = to_array(path)
        view = render(uploaded_file)
        return path, array, view

    return (handle_upload,)


@app.cell
def _(DataLoader, mo, os, tempfile):
    def save_to_tmp(upload):
        if not upload.value:
            return None
        file = upload.value[0]
        temp_dir = tempfile.mkdtemp(prefix="marimo_uploads_")
        safe_filename = os.path.basename(file.name)
        file_path = os.path.join(temp_dir, safe_filename)
        with open(file_path, "wb") as f:
            f.write(file.contents)
        return file_path

    def to_array(filepath):
        if not filepath:
            return None
        loader = DataLoader()
        return loader.load(filepath)

    def render(file):
        if not file.value:
            return mo.md(
                """
                <div style="display: flex; align-items: center; justify-content: center; height: 300px; border: 2px dashed #cbd5e1; border-radius: 8px; color: #64748b; text-align: center; background-color: #f8fafc; padding: 20px;">
                    No file available.<br/>Upload an image above to view it here.
                </div>
                """
            )
        img_data = file.value[0].contents
        return mo.image(
            src=img_data,
            style={"max-width": "100%", "max-height": "400px", "object-fit": "contain"},
        )

    return render, save_to_tmp, to_array


@app.cell
def _(image_view, mask_view, mo):
    mo.vstack(
        [mo.md("# Uploaded Image"), mo.hstack([image_view, mask_view], justify="start")]
    )


@app.cell
def _(HabitatsExtractor, image_path, mask_path):
    def extract_habitats(habitats_number, algorithm):
        if not image_path or not mask_path:
            return
        n = habitats_number if habitats_number else 2
        algorithm = algorithm if algorithm else "kmeans"

        extractor = HabitatsExtractor()
        return extractor.execute(image_path, mask_path, n, algorithm)

    return (extract_habitats,)


@app.cell
def _(Image, np):
    def stack_habitats(image_array, habitats):
        base = Image.fromarray(image_array).convert("RGBA")
        alpha = 200
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for overlay, color in zip(habitats, colors):
            layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
            solid_color = Image.new("RGBA", base.size, color + (alpha,))
            mask = Image.fromarray(overlay).convert("L")
            layer.paste(solid_color, (0, 0), mask=mask)
            base = Image.alpha_composite(base, layer)
        return np.array(base.convert("RGB"))

    return (stack_habitats,)


@app.cell
def _(mo):
    algorithm = mo.ui.dropdown(
        ["kmeans", "agglomerative"], label="Algorithm", value="kmeans"
    )
    habitats_number = mo.ui.number(start=1, stop=6, label="Habitats number")
    return algorithm, habitats_number


@app.cell
def _(
    algorithm,
    extract_habitats,
    habitats_number,
    image_array,
    stack_habitats,
):
    habitats = extract_habitats(habitats_number.value, algorithm.value)
    final_image = stack_habitats(image_array, habitats)
    return (final_image,)


@app.cell
def _(algorithm, final_image, habitats_number, mo):
    mo.hstack(
        [
            mo.vstack([mo.md("# Habitats"), habitats_number, algorithm]),
            mo.image(final_image, rounded=True),
        ]
    )


if __name__ == "__main__":
    app.run()
