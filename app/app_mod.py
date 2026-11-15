import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

@app.cell
def imports():
    import os
    import tempfile
    import marimo as mo
    import numpy as np
    import pandas as pd
    from PIL import Image

    from pyhabitats.image_loader import ImageLoader
    from pyhabitats.habitats_extractor import HabitatsExtractor
    from pyhabitats.features_extractor import FeaturesExtractor


    return (
        FeaturesExtractor,
        HabitatsExtractor,
        Image,
        ImageLoader,
        mo,
        np,
        os,
        tempfile,
    )


@app.cell
def ui_elements(mo):
    mo.md("# Detecting and analysis of Radiomic Habitats")

    image_upload = mo.ui.file(label="Upload Image (.nii / .dcm)", kind="area", multiple=True)
    mask_upload = mo.ui.file(label="Upload Mask", kind="area")

    binarization_method = mo.ui.dropdown(["positive", "threshold"], label="Binarization Methon", value="positive")
    algorithm = mo.ui.dropdown(["kmeans", "agglomerative"], label="Algorithm", value="kmeans")
    habitats_number = mo.ui.number(start=1, stop=6, value=2, label="Number of Habitats")

    feature_classes = mo.ui.multiselect(
        options=["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape", "shape2D"], 
        value=["firstorder", "glcm"], 
        label="Select Features"
    )

    return (
        algorithm,
        binarization_method,
        feature_classes,
        habitats_number,
        image_upload,
        mask_upload,
    )


@app.cell
def layout_inputs(
    algorithm,
    binarization_method,
    feature_classes,
    habitats_number,
    image_upload,
    mask_upload,
    mo,
):
    mo.vstack([
        mo.md("# Input Parameters"),
        mo.hstack([image_upload, mask_upload]),
        mo.hstack([binarization_method, algorithm, habitats_number]),
        feature_classes
    ])


@app.cell
def helpers(Image, np, os, tempfile):
    def save_to_tmp(file_obj):
        temp_dir = tempfile.mkdtemp(prefix="marimo_uploads_")
        safe_filename = os.path.basename(file_obj.name)
        file_path = os.path.join(temp_dir, safe_filename)
        with open(file_path, "wb") as f:
            f.write(file_obj.contents)
        return file_path

    def stack_habitats(image_array, habitats, z_index=None):
        if image_array.ndim >= 3:
            idx = z_index if z_index is not None else image_array.shape[2] // 2
            slice_img = image_array[:, :, idx]
        else:
            slice_img = image_array

        base_norm = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8) * 255).astype(np.uint8)
        base = Image.fromarray(base_norm).convert("RGBA")
    
        alpha = 150
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
        for i, overlay in enumerate(habitats):
            color = colors[i % len(colors)]
            if overlay.ndim >= 3:
                idx = z_index if z_index is not None else overlay.shape[2] // 2
                slice_overlay = overlay[:, :, idx]
            else:
                slice_overlay = overlay
            
            layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
            solid_color = Image.new("RGBA", base.size, color + (alpha,))
            mask_img = Image.fromarray((slice_overlay * 255).astype(np.uint8)).convert("L")
            layer.paste(solid_color, (0, 0), mask=mask_img)
            base = Image.alpha_composite(base, layer)
        
        return np.array(base.convert("RGB"))

    return save_to_tmp, stack_habitats


@app.cell
def processing(
    FeaturesExtractor, HabitatsExtractor, ImageLoader, 
    algorithm, binarization_method, feature_classes, habitats_number, 
    image_upload, mask_upload, np, save_to_tmp
):
    features_df = None
    habitats_list = None
    img_array = None
    loaded_images = []
    
    if image_upload.value and mask_upload.value:
        loader = ImageLoader()
        
        mask_path = save_to_tmp(mask_upload.value[0])
        mask_array = loader.load_image(mask_path)
        
        for file_info in image_upload.value:
            img_path = save_to_tmp(file_info)
            loaded_images.append(loader.load_image(img_path))
            
        if len(loaded_images) > 1:
            stacked_image = np.stack(loaded_images, axis=-1)
        else:
            stacked_image = loaded_images[0]
            
        img_array = loaded_images[0] 
        
        hab_extractor = HabitatsExtractor()
        habitats_list = hab_extractor.execute(
            image_input=stacked_image,
            mask_input=mask_array, 
            habitats_number=habitats_number.value, 
            algorithm=algorithm.value,
            binarization_method=binarization_method.value
        )
        
        combined_mask = np.zeros_like(mask_array, dtype=np.uint8)
        for idx, hab_mask in enumerate(habitats_list):
            combined_mask[hab_mask > 0] = idx + 1
            
        feat_extractor = FeaturesExtractor()
        
        if feature_classes.value:
            feat_extractor._extractor.disableAllFeatures()
            
            for f_class in feature_classes.value:
                feat_extractor._extractor.enableFeatureClassByName(f_class)
                
        features_df = feat_extractor.execute(img_array, combined_mask)
        
    return features_df, habitats_list, img_array, loaded_images


@app.cell
def dynamic_slider(img_array, mo):
    slice_slider = None
    if img_array is not None and img_array.ndim >= 3:
        max_z = img_array.shape[2] - 1
        slice_slider = mo.ui.slider(start=0, stop=max_z, value=max_z // 2, step=1, label="Single layer")
    return (slice_slider,)


@app.cell
def visualization(
    features_df,
    habitats_list,
    img_array,
    mo,
    slice_slider,
    stack_habitats,
):
    if img_array is not None and habitats_list is not None and features_df is not None:
    
        z_idx = slice_slider.value if slice_slider is not None else None
        vis_img = stack_habitats(img_array, habitats_list, z_index=z_idx)
    
        csv_bytes = features_df.to_csv(index=False).encode('utf-8')
        download_btn = mo.download(data=csv_bytes, filename="habitats_features.csv", label="Download CSV")
    
        out = mo.vstack([
            mo.md("# Clusterization Results"),
            slice_slider if slice_slider is not None else mo.md("*2D Image*"),
            mo.image(vis_img, rounded=True, style={"max-height": "400px"}),
            mo.md("# Radiomic Features"),
            download_btn,
            mo.ui.table(features_df, selection=None)
        ])
    else:
        out = mo.md("*Upload image and mask to get analysis*")
    
    out



if __name__ == "__main__":
    app.run()
