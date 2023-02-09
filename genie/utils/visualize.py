import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_protein_coords(coords, figsize=(500, 400), title=''):
    """
    Plot protein coordinates in 3D

    Parameters
    ----------
    coords : np.ndarray (N, 3)
        Ca coordinates of the protein
    figsize : tuple
        Size of the plot (width, height)
    title : str
        Title of the plot

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure object
    """

    nodes = go.Scatter3d(
    x=coords[:, 0],
    y=coords[:, 1],
    z=coords[:, 2],
    mode="lines+markers",
    marker={
        "symbol": "circle",
        "color": [plt.cm.plasma(i / coords.shape[0]) for i in range(coords.shape[0])],
        "size": 4,
        "opacity": 0.7,
    },
    line={
        "color": [plt.cm.plasma(i / coords.shape[0]) for i in range(coords.shape[0])],
        "width": 3}
    )

    axis = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title="",
    )

    return go.Figure(
            data=[nodes],
            layout=go.Layout(
                title=title,
                width=figsize[0],
                height=figsize[1],
                showlegend=False,
                scene=dict(
                    xaxis=dict(axis),
                    yaxis=dict(axis),
                    zaxis=dict(axis),
                ),
                margin=dict(t=0),
            ),
    )


def plot_structure_diffusion_animation(ts, animation_steps=10, figsize=(500, 400), title=''):
    '''
    Animation of Ca coordinates over timesteps.

    Parameters
    ----------
    ts : np.ndarray (T, N, 3)
        Ca coordinates of the protein over timesteps
    animation_steps : int
        Number of timesteps to skip between frames
    figsize : tuple
        Size of the plot (width, height)
    title : str
        Title of the plot

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure object
    '''
    
    def plot_protein(data):
        return go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode="lines+markers",
            marker={
                "symbol": "circle",
                "size": 4,
                "color": [plt.cm.plasma(i / ts.shape[1]) for i in range(ts.shape[1])],
                "opacity": 0.7,
            },
            line={
                "color": [plt.cm.plasma(i / ts.shape[1]) for i in range(ts.shape[1])],
                "width": 3},
            )

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    axis = dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title="",
        )

    frames = [go.Frame(data=[plot_protein(ts[k, :, :])],  name=f'frame{k}') for k in range(0, ts.shape[0], animation_steps)]

    fig = go.Figure(data=[plot_protein(ts[0, :, :])],
            frames=frames,
            layout=go.Layout(
                showlegend=False,
                        scene=dict(
                            xaxis=dict(axis),
                            yaxis=dict(axis),
                            zaxis=dict(axis),
                        ),
                        margin=dict(t=0)
            )
        )
        
    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    fig.update_layout(
            title=title,
            width=figsize[0],
            height=figsize[1],
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders
    )

    return fig