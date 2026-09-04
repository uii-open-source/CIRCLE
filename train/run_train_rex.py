import torch

from model.prompt_seg.rex_segmentation import RexGroundingCTSeg
from train.rex_trainer import RexSegTrainer


def print_trainable_parameters(model):
    """Print the ratio of trainable vs total parameters."""
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    ratio = 100 * trainable / total if total > 0 else 0
    print(f"trainable params: {trainable} || all params: {total} || trainable%: {ratio:.4f}")


def main():
    # ---------- User configuration ----------

    # --- Model / training hyperparameters ---
    # text embedding feature dim.  Must match the pre-computed .pt files in
    # `dataset_embed_dir`.  2560 = Qwen3-Embedding-4B last hidden.
    model_text_dim = 2560
    vision_pretrained = '/path/to/vision_encoder.bin'  # CIRCLE image encoder
    resume_checkpoint = ''

    freeze_image_encoder = False

    results_folder = '/path/to/results_folder'

    batch_size = 3
    lr = 3e-4
    max_grad_norm = 1.0
    num_train_steps = 200001
    num_workers = 6
    save_results_every = 1500
    save_model_every = 500

    # --- Dataset paths (explicitly forwarded to RexGroundingCTDataset) ---
    dataset_center_csv = \
        '/path/to/lung_center.csv'
    dataset_json_path = \
        '/path/to/MICCAI_challenge_dataset.json'
    dataset_image_dir = \
        '/path/to/image_folder'
    dataset_mask_dir = \
        '/path/to/segmentations_split/'
    dataset_embed_dir = \
        '/path/to/text_emb_4b'
    dataset_split = 'train'
    # ----------------------------------------

    model = RexGroundingCTSeg(
        text_dim=model_text_dim,
        vision_pretrained=vision_pretrained,
    )

    # Optionally freeze the image encoder (train only decoder)
    if freeze_image_encoder:
        for p in model.visual_encoder.parameters():
            p.requires_grad = False
        print('[INFO] Image encoder frozen; only segmentation decoder is trained.')

    # Resume from a previous segmentation checkpoint if provided
    if resume_checkpoint:
        state_dict = torch.load(resume_checkpoint, map_location='cpu')
        msg = model.load_state_dict(state_dict, strict=True)
        print(f'[INFO] Loaded resume checkpoint: {msg}')

    print_trainable_parameters(model)

    # Build trainer and start training
    trainer = RexSegTrainer(
        model,
        # --- dataset config ---
        dataset_center_csv=dataset_center_csv,
        dataset_json_path=dataset_json_path,
        dataset_image_dir=dataset_image_dir,
        dataset_mask_dir=dataset_mask_dir,
        dataset_embed_dir=dataset_embed_dir,
        dataset_split=dataset_split,
        # --- optimization / persistence ---
        batch_size=batch_size,
        lr=lr,
        max_grad_norm=max_grad_norm,
        num_train_steps=num_train_steps,
        num_workers=num_workers,
        save_results_every=save_results_every,
        save_model_every=save_model_every,
        results_folder=results_folder,
    )

    trainer.train()


if __name__ == '__main__':
    main()
