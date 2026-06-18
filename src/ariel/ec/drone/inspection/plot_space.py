import os, pickle
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file, save, curdoc
from bokeh.layouts import gridplot, column, row
from bokeh.io import export_png
from bokeh.models import ColorBar, LinearColorMapper, ColumnDataSource, CustomJS, TapTool, Div, HoverTool, BasicTicker, PrintfTickFormatter
from bokeh.transform import linear_cmap
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
from matplotlib import gridspec
from matplotlib.pyplot import hexbin

from ariel.ec.drone.inspection.drone_visualizer import DroneVisualizer

def generate_individual_images(population_data, fitnesss_data, output_folder, include_motor_orientation=True, twod=True):
    """
    Generates and saves images for each individual in the population data.
    
    Args:
        population_data (list): List of individuals in the population.
        output_folder (str): Folder to save the images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for gen_idx, generation in enumerate(population_data):
        for ind_idx, individual in enumerate(generation):
            # Generate the image (this is just a placeholder, replace with actual image generation)
            
            visualizer = DroneVisualizer()
            if twod:
                fig, ax = visualizer.plot_2d(individual, title=f"Gen {gen_idx}: Fitness {fitnesss_data[gen_idx, ind_idx]:.2f}")
            else:
                fig, ax = visualizer.plot_3d(individual, title=f"Gen {gen_idx}: Fitness {fitnesss_data[gen_idx, ind_idx]:.2f}")
            image_path = os.path.join(output_folder, f"gen{gen_idx}_ind{ind_idx}.png")
            plt.savefig(image_path)
            plt.close()

def plot_space_interactive(data, fitness, param_names=None, save_dir=None, density=True):
    ngen, pop_size, nparams = data.shape
    all_individuals = data.reshape(ngen * pop_size, nparams)

    image_dir = save_dir+'html/' + 'images/'
    dir = save_dir+'html/'
    print(image_dir)
    print(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
        

    plots = []
    for i in range(nparams-1):
        rows = []
        for j in range(nparams):
            if i < j:
                x = all_individuals[:, i]
                y = all_individuals[:, j]
                
                slope, intercept = np.polyfit(x, y, 1)
                trendline_y = [slope * xi + intercept for xi in x]

                # Interpolate the fitness data

                if density:
                    xy = np.vstack([x, y])
                    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
                    xi, yi = np.meshgrid(xi, yi)
                    
                    try:
                        zi = gaussian_kde(xy)(np.vstack([xi.flatten(), yi.flatten()]))
                    except:
                        zi = np.zeros(xi.shape)
                
                else:
                    grid_x, grid_y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
                    points = np.vstack([x, y]).T
                    values = fitness.flatten()
                    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
                
                    # Handle NaN values in the interpolated grid
                    grid_z = np.nan_to_num(grid_z, nan=np.nanmin(values))
                
                # Clear the current document
                curdoc().clear()
                
                # Create a figure
                p = figure(title=f'Scatter Plot of {param_names[i]} vs {param_names[j]}' if param_names else f'Scatter Plot of Param {i} vs Param {j}',
                           x_axis_label=param_names[i] if param_names else f'Param {i}',
                           y_axis_label=param_names[j] if param_names else f'Param {j}',
                           width=900, height=700, tools="tap,hover")
                
                # Plot background
                if density:
                    # Plot density background
                    density_mapper = LinearColorMapper(palette="Magma256", low=zi.min(), high=zi.max())
                    p.image(image=[zi.reshape(xi.shape)], x=x.min(), y=y.min(), dw=x.max()-x.min(), dh=y.max()-y.min(), color_mapper=density_mapper)
                    
                    # Add color bar for density
                    density_color_bar = ColorBar(color_mapper=density_mapper, ticker=BasicTicker(), 
                                                formatter=PrintfTickFormatter(format="%d"), 
                                                label_standoff=12, border_line_color=None, location=(0,0),
                                                title="Density")
                    p.add_layout(density_color_bar, 'right')
                else:
                    # Plot Fitness background
                    fitness_mapper = LinearColorMapper(palette="Magma256", low=np.nanmin(fitness), high=np.nanmax(fitness))
                    p.image(image=[grid_z.T], x=x.min(), y=y.min(), dw=x.max()-x.min(), dh=y.max()-y.min(), color_mapper=fitness_mapper)
                
                    # Add color bar for fitness
                    fitness_color_bar = ColorBar(color_mapper=fitness_mapper, ticker=BasicTicker(), 
                                                formatter=PrintfTickFormatter(format="%d"), 
                                                label_standoff=12, border_line_color=None, location=(0,0),
                                                title="Fitness")
                    p.add_layout(fitness_color_bar, 'right')

                # Color points by their index
                images = [os.path.abspath(os.path.join(image_dir, f'gen{g}_ind{i}.png')) for g in range(ngen) for i in range(pop_size)]
                source = ColumnDataSource(data={'x': x, 'y': y, 
                                                'gen': np.repeat(np.arange(ngen), pop_size), 
                                                'ind': np.tile(np.arange(pop_size), ngen), 
                                                'fitness': fitness.flatten(),
                                                'images': images})
                # Create a color mapper for generations
                gen_mapper = LinearColorMapper(palette="Viridis256", low=0, high=ngen-1)
                p.scatter('x', 'y', color={'field': 'gen', 'transform': gen_mapper}, source=source)
                # p.line(x, trendline_y, line_width=2, color="red", legend_label="Trendline")
                # Add color bar for generations
                gen_color_bar = ColorBar(color_mapper=gen_mapper, ticker=BasicTicker(), 
                                        formatter=PrintfTickFormatter(format="%d"), 
                                        label_standoff=12, border_line_color=None, location=(0,0),
                                        title="Generation")
                p.add_layout(gen_color_bar, 'right')
                
                hover = p.select(dict(type=HoverTool))
                hover.tooltips = """
                <div>
                    <div>
                        <span style="font-size: 10px;">x:@x, y:@y, F:@fitness, G:@gen</span>
                    </div>
                </div>
                """

                # Create a Div to display the image
                div = Div(width=700, height=700)
                
                # CustomJS callback to update the image in the Div
                callback = CustomJS(args=dict(source=source, image_div=div), code="""
                    const indices = cb_data.index.indices;
                    if (indices.length > 0) {
                        const index = indices[0];  // Select the first index
                        const image_url = source.data['images'][index];
                        image_div.text = '<img src="file://' + image_url + '" style="width:100%; height:100%;" />';
                    } else {
                        image_div.text = '';
                    }
                """)

                # Add the callback to the HoverTool
                hover.callback = callback
                
                layout = row(p, div)
                rows.append(layout)
                
                # Save each pair in a separate figure
                output_file(f'{dir}scatter_{param_names[i]}_{param_names[j]}.html')
                save(layout)
                show(layout)
        plots.append(row)

def plot_space(data, param_names=None, save_name=None, fitnesses=None, points=True, interpolation_method='tricontourf', point_size=5):
    print(data.shape)
    if fitnesses is not None:
        print(fitnesses.shape)
    ngen, pop_size, nparams = data.shape
    all_individuals = data.reshape(ngen * pop_size, nparams)

    # Create a grid spec for the combined figure and color bars
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nparams-1, nparams-1)  # Adjust grid spec to remove unused rows

    axes = np.empty((nparams-1, nparams-1), dtype=object)
    for i in range(nparams-1):
        for j in range(nparams):
            if i < j:  # Only plot above the diagonal
                ax = fig.add_subplot(gs[i, j-1])  # Adjust indices to fit the new grid spec
                axes[i, j-1] = ax
                x = all_individuals[:, i]
                y = all_individuals[:, j]
                xy = np.vstack([x, y])
                if fitnesses is not None:
                    fitnesses_flat = fitnesses.reshape(ngen * pop_size)
                    filtered = np.logical_or(~np.any(np.isnan(xy), axis=0), np.isnan(fitnesses_flat))
                else:
                    filtered = ~np.any(np.isnan(xy), axis=0)

                x = x[filtered]
                y = y[filtered]
                if fitnesses is not None:
                    fitnesses_filtered = fitnesses_flat[filtered]    
                xy = xy[:,filtered]

                # assert len(x) == len(y) == len(fitnesses_filtered)
                if fitnesses is not None:
                    if interpolation_method == 'tricontourf':
                        try:
                            density_plot = ax.tricontourf(x, y, fitnesses_filtered, cmap='viridis')
                        except:
                            pass
                    elif interpolation_method == 'hexbin':
                        hexd = ax.hexbin(x, y, C=fitnesses_filtered, gridsize=20, cmap='viridis')
                        ax.set(xlim=(min(x),max(x)), ylim=(min(y),max(y)))
                    elif interpolation_method == 'linear':
                        try:
                            xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
                            xi, yi = np.meshgrid(xi, yi)
                            zi = griddata((x, y), fitnesses_filtered, (xi, yi), method='linear')
                            density_plot = ax.contourf(xi, yi, zi, levels=100, cmap='viridis')
                        except:
                            pass
                            
                    else:
                        print('Interpolation method not recognized.')
                        return
                else:
                    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
                    xi, yi = np.meshgrid(xi, yi)
                    zi = gaussian_kde(xy)(np.vstack([xi.flatten(), yi.flatten()]))
                    density_plot = ax.contourf(xi, yi, zi.reshape(xi.shape), levels=100, cmap='viridis')
                if points:
                    colors = np.repeat(np.arange(ngen).reshape((1,ngen)),pop_size,axis=1)[0]
                    colors = colors[filtered]
                    scatter = ax.scatter(x, y, c=colors, cmap='coolwarm', edgecolor='k', s=point_size)

                if param_names is not None:
                    ax.set_xlabel(param_names[i])
                    ax.set_ylabel(param_names[j])
                else:
                    ax.set_xlabel(f'Param {i}')
                    ax.set_ylabel(f'Param {j}')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Plot each pair in a separate figure
                fig_sep, ax_sep = plt.subplots(figsize=(8, 6))
                if fitnesses is None:
                    density_plot_sep = ax_sep.contourf(xi, yi, zi.reshape(xi.shape), levels=100, cmap='viridis')
                else:
                    if interpolation_method == 'tricontourf':
                        try:
                            density_plot_sep = ax_sep.tricontourf(x, y, fitnesses_filtered, cmap='viridis')
                        except:
                            pass
                    elif interpolation_method == 'linear':
                        xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
                        xi, yi = np.meshgrid(xi, yi)
                        zi = griddata((x, y), fitnesses_filtered, (xi, yi), method='linear')
                        try:
                            density_plot_sep = ax_sep.contourf(xi, yi, zi, levels=100, cmap='viridis')
                        except:
                            pass

                    elif interpolation_method == 'hexbin':
                        hexd_sep = ax_sep.hexbin(x, y, C=fitnesses_filtered, gridsize=20, cmap='viridis')
                        ax_sep.set(xlim=(min(x),max(x)), ylim=(min(y),max(y)))
                if points:
                    zi = np.repeat(np.arange(ngen).reshape((1,ngen)),pop_size,axis=1)[0]
                    zi = zi[filtered]
                    scatter_sep = ax_sep.scatter(x, y, c=zi if fitnesses is not None else colors, cmap='coolwarm', edgecolor='k', s=point_size)
                if param_names is not None:
                    ax_sep.set_xlabel(param_names[i])
                    ax_sep.set_ylabel(param_names[j])
                    ax_sep.set_title(f'Scatter Plot of {param_names[i]} vs {param_names[j]}')
                else:
                    ax_sep.set_xlabel(f'Param {i}')
                    ax_sep.set_ylabel(f'Param {j}')
                    ax_sep.set_title(f'Scatter Plot of Param {i} vs Param {j}')
                
                # Add color bars for the separate figure on the right
                fig_sep.subplots_adjust(right=0.7)  # Adjust the right margin to make space for color bars
                if fitnesses is None:
                    cbar_ax1_sep = fig_sep.add_axes([0.72, 0.1, 0.03, 0.8])
                    fig_sep.colorbar(density_plot_sep, cax=cbar_ax1_sep, orientation='vertical', label='Density')
                cbar_ax2_sep = fig_sep.add_axes([0.82, 0.1, 0.03, 0.8])

                if points:
                    fig_sep.colorbar(scatter_sep, cax=cbar_ax2_sep, orientation='vertical', label='Fitness' if fitnesses is not None else 'Index')
                # elif:
                #     fig_sep.colorbar(hexd_sep, cax=cbar_ax2_sep, orientation='vertical', label='Fitness' if fitnesses is not None else 'Index')
                if save_name is not None:
                    plt.savefig(f'{save_name}_scatter_{param_names[i]}_{param_names[j]}.png')

    # Add color bars at the bottom for the combined figure
    fig.subplots_adjust(bottom=0.2)
    cbar_ax1 = fig.add_axes([0.55, 0.1, 0.35, 0.02])
    if points:
        cbar_ax2 = fig.add_axes([0.1, 0.1, 0.35, 0.02])

    if fitnesses is None:
        fig.colorbar(density_plot, cax=cbar_ax1, orientation='horizontal', label='Density')
    else:
        if interpolation_method == 'hexbin':
            fig.colorbar(hexd, cax=cbar_ax1, orientation='horizontal', label='Fitness')
        else:
            pass

    if points:
        fig.colorbar(scatter, cax=cbar_ax2, orientation='horizontal', label='Generation')
    if save_name is not None:
        fig.savefig(f'{save_name}_scatter_combined.png')
    # make animation of points over time 
    # plot 3d scatter of points, color represents generation
    # plot animated 3d scatter of pointsx

    # other options for plots: PCA, t-SNE, UMAP, MDS, LDA, Isomap, LLE, Laplacian Eigenmaps, Hessian Eigenmaps, Diffusion Maps, Spectral Embedding, Neighborhood Components Analysis

def plot_specific_morphological_descriptors_space(population_data, md_funcs, md_names, save_name=None, fitnesses=None, points=True, interpolation_method='tricontourf', point_size=5):
    
    # Get the data for the morphological descriptors
    data = np.empty((population_data.shape[0], population_data.shape[1], len(md_funcs)))
    for i, md_func in enumerate(md_funcs):
        data[:,:,i] = md_func(population_data)
    
    # Save the data to a file
    if save_name is not None:
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        try:
            np.save(save_name+".npy", data)
        except:
            with open(save_name+".pkl", 'wb') as f:
                pickle.dump(data, f)
    # Plot the data
    plot_space(data, param_names=md_names, save_name=save_name, fitnesses=fitnesses, points=points, interpolation_method=interpolation_method, point_size=point_size)


def plot_morphological_descriptors_space(population_data, save_name=None, twod=True, hover=False, fitnesses=None, points=True):

    from ariel.ec.drone.inspection.morphological_descriptors.area import compute_area
    from ariel.ec.drone.inspection.morphological_descriptors.avr_arm_length import compute_avr_arm_length
    from ariel.ec.drone.inspection.morphological_descriptors.mass import compute_total_mass
    from ariel.ec.drone.inspection.morphological_descriptors.num_arms import compute_num_arms
    from ariel.ec.drone.inspection.morphological_descriptors.proportion import compute_proportion
    from ariel.ec.drone.inspection.morphological_descriptors.central_symmetry import compute_symmetry
    from ariel.ec.drone.inspection.morphological_descriptors.var_arm_length import compute_var_arm_length
    from ariel.ec.drone.inspection.morphological_descriptors.volume import compute_volume
    from ariel.ec.drone.inspection.morphological_descriptors.hovering_info import compute_hovering_info
    
    num_arms_data = compute_num_arms(population_data)
    mass_data = compute_total_mass(population_data)
    avr_arm_length_data = compute_avr_arm_length(population_data)
    var_arm_length_data = compute_var_arm_length(population_data)
    proportion_data = compute_proportion(population_data)
    symmetry_data = compute_symmetry(population_data)
    if twod:
        space_data = compute_area(population_data)
    else:
        space_data = compute_volume(population_data)
    if hover:
        hovering_data = compute_hovering_info(population_data)
        can_hover = hovering_data[:,0]
        weight_to_thrust = hovering_data[:,1]
        input_cost = hovering_data[:,2]
        rank_controlability = hovering_data[:,3]
        controllability = hovering_data[:,4]

        if len(population_data.shape) == 4:
            data = np.concatenate([np.expand_dims(num_arms_data, axis=2),
                                np.expand_dims(mass_data, axis=2),
                                np.expand_dims(avr_arm_length_data, axis=2),
                                np.expand_dims(var_arm_length_data, axis=2),
                                np.expand_dims(proportion_data, axis=2),
                                np.expand_dims(symmetry_data, axis=2),
                                np.expand_dims(space_data, axis=2),
                                np.expand_dims(can_hover, axis=2),
                                np.expand_dims(weight_to_thrust, axis=2),
                                np.expand_dims(input_cost, axis=2),
                                np.expand_dims(rank_controlability, axis=2),
                                np.expand_dims(controllability, axis=2)], axis=2)
        elif len(population_data.shape) == 3:
            data = np.expand_dims(np.concatenate([np.expand_dims(num_arms_data, axis=1),
                                np.expand_dims(mass_data, axis=1),
                                np.expand_dims(avr_arm_length_data, axis=1),
                                np.expand_dims(var_arm_length_data, axis=1),
                                np.expand_dims(proportion_data, axis=1),
                                np.expand_dims(symmetry_data, axis=1),
                                np.expand_dims(space_data, axis=1),
                                np.expand_dims(can_hover, axis=1),
                                np.expand_dims(weight_to_thrust, axis=1),
                                np.expand_dims(input_cost, axis=1),
                                np.expand_dims(rank_controlability, axis=1),
                                np.expand_dims(controllability, axis=1)], axis=1), axis=0)

        param_names = ['Number of Arms', 'Total Mass', 'Avr Arm Length', 'Variance of Arm lengths', 
                    'Proportion', 'Symmetry', 'Volume', 'Can Hover', 'Weight to Thrust Ratio', 'Input Cost', 
                    'Rank Controlability', 'Controllability']
    else:
        if len(population_data.shape) == 4:
            data = np.concatenate([np.expand_dims(num_arms_data, axis=2),
                                np.expand_dims(mass_data, axis=2),
                                np.expand_dims(avr_arm_length_data, axis=2),
                                np.expand_dims(var_arm_length_data, axis=2),
                                np.expand_dims(proportion_data, axis=2),
                                np.expand_dims(symmetry_data, axis=2),
                                np.expand_dims(space_data, axis=2)], axis=2)
        elif len(population_data.shape) == 3:
            data = np.expand_dims(np.concatenate([np.expand_dims(num_arms_data, axis=1),
                                np.expand_dims(mass_data, axis=1),
                                np.expand_dims(avr_arm_length_data, axis=1),
                                np.expand_dims(var_arm_length_data, axis=1),
                                np.expand_dims(proportion_data, axis=1),
                                np.expand_dims(symmetry_data, axis=1),
                                np.expand_dims(space_data, axis=1)], axis=1), axis=0)

        param_names = ['Number of Arms', 'Total Mass', 'Avr Arm Length', 'Variance of Arm lengths', 
                    'Proportion', 'Symmetry', 'Volume']
    
    if twod:
        param_names[6] = 'Area'
    else:
        param_names[6] = 'Volume'

    plot_space(data, param_names=param_names, save_name=save_name, points=points, fitnesses=fitnesses)


def show_html_files_in_directory(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter to include only HTML files
    html_files = [f for f in files if f.endswith('.html')]
    
    for html_file in html_files:
        # Construct the full file path
        file_path = os.path.join(directory, html_file)
        
        # Output the file
        output_file(file_path)
        
        # Create a simple plot (or load your plot from the file)
        p = figure(title="HTML File: " + html_file)
        
        # Show the plot
        show(p)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm

def plot_space_time(population_data, evaluator, xlabel="Area", ylabel="Proportion", zlabel="Mass"):
    ngens, pop_size, narms, nparms = population_data.shape
    points = np.empty((ngens, pop_size, 3))
    for i, pop in enumerate(population_data):
        points[i] = evaluator.get_points_from_pop(pop)

    # Function to update the scatter plot and text
    def update(num, data, scatter, text, colormap):
        # Accumulate points up to the current frame
        current_data = data[:num+1].reshape(-1, 3)  # Combine all points from previous frames
        colors = colormap(np.linspace(0, 1, num+1).repeat(pop_size))  # Color gradient over frames
        
        scatter._offsets3d = (current_data[:, 0], current_data[:, 1], current_data[:, 2])
        scatter.set_color(colors)  # Apply the color map
        
        # Update the text to show the current generation/frame
        text.set_text(f'Generation: {num+1}')
        
        return scatter, text

    # Set up the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits (optional, depending on your data)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    # Initialize scatter plot with no points (start with an empty plot)
    scatter = ax.scatter([], [], [])

    # Add initial text for generation
    text = fig.text(0.5, 0.95, '', ha='center', va='top', fontsize=14, color='black')

    # Create a colormap (using 'viridis' or any other colormap of your choice)
    colormap = cm.get_cmap('plasma')

    # Create the animation, accumulating points and changing color with each frame
    ani = FuncAnimation(fig, update, frames=ngens, 
                        fargs=(points, scatter, text, colormap), interval=200, blit=False)

    ani.save('animation.mp4', writer='ffmpeg', fps=10)
    # Show the plot
    plt.show()