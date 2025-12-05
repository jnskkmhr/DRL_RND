import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb


# matplotlib global settings
mpl.rcParams.update({
    "pdf.fonttype": 42,            # editable text in Adobe Illustrator
    "ps.fonttype": 42,             # editable text in Adobe Illustrator
    
    "figure.dpi": 200,
    "savefig.dpi": 600,              # very high-res when saving
    "savefig.format": "pdf",         # vector output preferred
    # Fonts
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # IEEE default
    # Axes
    "axes.linewidth": 1.5,
    # Lines
    "lines.linewidth": 2,
})


def plot(wandb_username: str, # WANDB_USERNAME not in env
         wandb_project: str = "rnd_rl",
         safety_experiment: bool = False,
         indices: list[int] | None = None,
         labels: list[str] | None = None,
         title: str | None = None,
         double_column: bool = True,
         epochs: int = 250,
         top_ylims: list[float] | None = None
         ) -> plt.figure:
    '''
    Note: this function is NOT TESTED due to current connection issued to wandb.

    plots 2 subplots. safety_experiemnt = False: plots combined and extrinsice rewards; 
    safety_experiemnt = True: plots constraint violations and extrinsic rewards.
    
    :param wandb_username: Your wandb username 
    :type wandb_username: str
    :param wandb_project: Your wandb project name. By default in this project, is "rnd_rl". 
    :type wandb_project: str
    :param safety_experiment: Plots safety violation constraints if they exists in the experiments.
    :type safety_experiment: bool
    :param indices: Indices of runs you want to plot. If not specified, will plot all runs.
    :type indices: list[int]
    :param labels: Labels of runs to display. 
    Should match the length of indices if specified. If not specified, will be the experiment names.
    :type labels: list[str] | None
    :param double_column: Whether the image fits IEEE format.
    :type double_column: bool
    :param epochs: changes xlim.
    :type epochs: int
    :param top_ylims: changes top ylim if specified
    :type top_ylims: list[float] | None
    '''
    # fetch data from wandb
    # wandb_username = os.environ["WANDB_USERNAME"]

    api = wandb.Api()
    project_name = wandb_username + "/" + wandb_project
    project = api.runs(project_name)

    # specify figure size

    if double_column: # see IEEE style guide
        fontsize=8
        fig, axes = plt.subplots(2,1, sharex=True, figsize=(3.5, 2.5)) # 2 rows
    else:
        fontsize=16
        fig, axes = plt.subplots(1,2, sharey=True, figsize=(10, 2)) # 2 cols
    
    n_runs = len(project) if indices is None else len(indices)

    for i in range(n_runs):
        idx = i if indices is None else indices[i] 
        run = project[idx]
        df = run.history(samples=1000, pandas=True)

        label = run.name if labels is None else labels[i]
        if safety_experiment:
            # Units: percentage of violations. Original data is ratio of violations.
            axes[0].plot(df['_step'], df['State/Cart_Position_Violation']*100, label=label)
            axes[1].plot(df['_step'], df['Extrinsic Reward'], label=label)
        else:
            axes[0].plot(df['_step'], df['Reward'], label=label)
            axes[1].plot(df['_step'], df['Extrinsic Reward'], label=label)

    if title is not None:
        # fig.suptitle(t=title, fontsize=fontsize)
        axes[0].set_title(label = title, fontsize=fontsize)


    for i in range(len(axes)):
        ax = axes[i]
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_xlim(0,epochs)
        ax.grid()

        if (top_ylims is not None) and (top_ylims[i] is not None):
            ax.set_ylim(top = top_ylims[i])

    if safety_experiment:
        axes[0].set_ylabel("Violations %", fontsize=fontsize)
    else:
        axes[0].set_ylabel("Combined", fontsize=fontsize)

    axes[1].set_ylabel("Extrinsic", fontsize=fontsize)

    axes[1].set_xlabel("Training Steps", fontsize=fontsize)
    if not double_column:  # the format in midterm
        axes[0].set_xlabel("Training Steps", fontsize=fontsize)


    # axes[1].legend(frameon=False)
    axes[1].legend(fontsize = 6)

    plt.tight_layout()
    # plt.savefig("training_rewards.pdf")
    plt.show()

    return fig