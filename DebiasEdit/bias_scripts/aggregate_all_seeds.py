import pandas as pd
import argparse
import os
import numpy as np

import matplotlib.pyplot as plt



base_results = {
     "edited_success": 0,
     "less_all_vs_more_all": 0,
     "less_vs_more_para": 14.75,
     "locality": 100
} 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_dir', default="llama")
    parser.add_argument('--model', default="llama2-7b")
    parser.add_argument('--seed_list', nargs='+')
    parser.add_argument('--alg_list', nargs='+')

    args = parser.parse_args()
    metrics_dir = f"DebiasEdit/results_{args.metrics_dir}/sequential/steroset/metrics"
    
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 15))
    axes = axes.flatten()
    for status_index, status in enumerate(["edited", "unedited"]):
        results = {}
        
        for alg in args.alg_list:#"MEND", "SERAC"]:
            all_data = []
            for seed in args.seed_list:
                
                filename = f'{alg}_{status}_results_seed{seed}.csv'
                try:
                    df = pd.read_csv(os.path.join(metrics_dir, filename),index_col=0)
                    all_data.append(df)
                except:
                    pass
            
            all_df = pd.concat(all_data)
            
            grouped = all_df.groupby(all_df.index)
            mean_df = grouped.mean()
            std_df = grouped.std()
            mean_df = mean_df.round(2)
            std_df = std_df.round(2)
            results[alg] = {}
            results[alg]["mean"] = mean_df
            results[alg]["std"] = std_df


           
            results[alg]["mean ± std"] = mean_df.astype(str) + ' ± ' + std_df.astype(str)
    
            print(results[alg]["mean ± std"])

            if not os.path.exists(f"final/{args.name}"):
                os.makedirs(f"final/{args.name}")
            results[alg]["mean ± std"].to_csv(f"final/{args.metrics_dir}/{alg}_{status}.csv")


        if not os.path.exists('plots'):
            os.makedirs('plots')
        if status == "edited":
            x_labels = [ '$2^0$', '$2^2$', '$2^4$', '$2^6$', '$2^8$', 'ALL']
        else:
            x_labels = [ '$2^0$', '$2^2$', '$2^4$', '$2^6$', '$2^8$']
        x = range(len(x_labels))

        for i, (index, row) in enumerate(results["FT"]["mean"].iterrows()):
            ax = axes[status_index * 4 + i]
            ax.axhline(y=base_results[index], linestyle='--', linewidth=2, label="Base Model")
        
            for key in results.keys():
                
                mean_value = results[key]["mean"].loc[index]
                std_value = results[key]["std"].loc[index]
                mean_value = mean_value
                std_value = std_value
                
                ly = [max(0, mean - std) for mean, std in zip(mean_value, std_value)]
                hy = [min(mean + std, 100) for mean, std in zip(mean_value, std_value)]
                if index == "less_vs_more_para":
                    label = "${GEN}_{forward}$"
                elif index == "less_all_vs_more_all":
                    label = "${GEN}_{backward}$"
                elif index == "edited_success":
                    label = "Success Rate"
                elif index == "locality":
                    label = "Knowledge Acc"
        
                ax.errorbar(x, mean_value, capsize=4, marker='o', label=key, markersize=5)
                ax.fill_between(x, ly, hy, alpha=0.2, zorder=0)

            ax.set_title(f'{label}$\\uparrow$', fontsize=30)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=25)
            ax.tick_params(axis='y', labelsize=25)
            ax.grid()
            
            # ax.legend(fontsize="large")  # 显示图例
    handles, labels =  axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(results.keys())+2, fontsize=30)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5, wspace=0.3)
    import matplotlib.lines as mlines

    subplot_height = fig.get_figheight() / nrows
    line_position = subplot_height / fig.get_figheight()

    line = mlines.Line2D([0, 1], [line_position+0.015, line_position+0.015], color='gray', linestyle='--', linewidth=1, transform=fig.transFigure, figure=fig)
    fig.lines.extend([line])

    title1 = 'Edited'
    title2 = 'Unedited'
    title_offset = 0.05

    fig.text(0.5, 1 - title_offset, title1, ha='center', va='center', fontsize=35, weight='bold')
    fig.text(0.5, 0.5 - 0.03, title2, ha='center', va='center', fontsize=35, weight='bold')

    plt.savefig(os.path.join('plots', f'all_metrics_{args.name}_{args.model}.pdf'), format="pdf")
    plt.close(fig)

if __name__ == "__main__":
    main()