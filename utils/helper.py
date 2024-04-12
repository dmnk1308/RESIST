import matplotlib.pyplot as plt
import wandb

def log_heatmaps(targets, preds, n_examples=20, n_res=4):
    # log qualitative results
    targets_case = targets.detach().cpu().numpy().squeeze().reshape(-1, n_res, 512, 512)
    preds_case = preds.detach().cpu().numpy().squeeze().reshape(-1, n_res, 512, 512)
    for i in range(n_examples):
        fig, axes = plt.subplots(n_res, 2, figsize=(10, 16))
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        for resistancy in range(n_res):
            axes[resistancy,0].imshow(targets_case[i,resistancy], vmin=0, vmax=0.7, cmap='coolwarm')
            axes[resistancy,0].axis('off')
            axes[resistancy,1].imshow(preds_case[i,resistancy], vmin=0, vmax=0.7, cmap='coolwarm')
            axes[resistancy,1].axis('off')
        # Add colorbar to the figure
        sm = plt.cm.ScalarMappable(cmap='coolwarm')
        sm.set_clim(0, 0.7)
        cbar = fig.colorbar(sm, cax=cbar_ax)        
        cbar.set_label('Specific Conductivity (S/m)')
        wandb.log({'Heatmap': fig})
        plt.close(fig)