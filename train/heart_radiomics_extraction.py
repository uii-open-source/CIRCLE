import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import warnings
warnings.filterwarnings('ignore')

def resample_mask_to_image(mask, image):
    """Resample mask to match the spatial properties of the image"""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampled_mask = resampler.Execute(mask)
    return resampled_mask

def extract_features(image_path, mask_path, extractor):
    """Extract radiomics features from a single image-mask pair"""
    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        
        # Check if image and mask have same dimensions
        if image.GetSize() != mask.GetSize():
            print(f"  Resampling mask to match image dimensions...")
            mask = resample_mask_to_image(mask, image)
        
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        result = extractor.execute(image, mask, label=1)
        
        features = {}
        for key, value in result.items():
            if not key.startswith('diagnostics'):
                features[key] = value
        return features
    except Exception as e:
        print(f"Error: {image_path} - {e}")
        return None

def main():
    # Configure paths
    image_dir = r"data\image"
    mask_dir = r"data\mask"
    output_csv = r"data\radiomics_features_RWMA.csv"
    
    # Configure PyRadiomics parameters
    params = {
        'minimumROIDimensions': 2,
        'minimumROISize': 1,
        'normalize': False,
        'label': 1,
    }
    
    settings = {
        'binWidth': 25,
        'normalize': False,
        'symmetricalGLCM': True,
    }
    
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.settings.update(settings)
    
    # Enable 7 feature classes, disable wavelet transforms
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('ngtdm')
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName('Original')
    
    # Get all sample folders
    folders = [f for f in os.listdir(image_dir) 
               if os.path.isdir(os.path.join(image_dir, f))]
    
    all_features = []
    processed, skipped = 0, 0
    
    for folder in sorted(folders):
        image_path = os.path.join(image_dir, folder, "CT.nii.gz")
        mask_path = os.path.join(mask_dir, folder, "heart_mask.nii.gz")
        
        if not all([os.path.exists(image_path), os.path.exists(mask_path)]):
            print(f"Skipped: {folder} (file missing)")
            skipped += 1
            continue
        
        print(f"Processing: {folder}")
        features = extract_features(image_path, mask_path, extractor)
        
        if features:
            all_features.append(features)
            processed += 1
        else:
            skipped += 1
    
    if not all_features:
        print("No features extracted.")
        return
    
    # Create DataFrame and save
    df = pd.DataFrame(all_features)
    
    # Get the list of successfully processed folders
    processed_folders = []
    for folder in sorted(folders):
        image_path = os.path.join(image_dir, folder, "CT.nii.gz")
        mask_path = os.path.join(mask_dir, folder, "heart_mask.nii.gz")
        if all([os.path.exists(image_path), os.path.exists(mask_path)]):
            processed_folders.append(folder)
    
    # Ensure we have the same number of processed folders and feature sets
    processed_folders = processed_folders[:len(all_features)]
    df.insert(0, 'image_name', processed_folders)
    
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    # Output statistics
    print(f"\nProcessed: {processed}, Skipped: {skipped}")
    print(f"Output shape: {df.shape}")
    print(f"Features saved to: {output_csv}")
    
    # Show feature names
    if len(df.columns) > 1:
        print(f"\nExtracted {len(df.columns)-1} features")
        print("First 10 features:")
        for i, col in enumerate(df.columns[1:11]):
            print(f"  {i+1:3}. {col}")

if __name__ == "__main__":
    main()
