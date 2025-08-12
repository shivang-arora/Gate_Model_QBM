import seaborn as sns
import matplotlib.pyplot as plt
import random
from pennylane import numpy as np
import matplotlib.colors as mcolors



def plot_pixel_dataset(data,CLUSTER):
    data=data

    x_values = np.arange(0,16,1)
    y_values=np.arange(0,16,1)

    c_values=[]
    for x in x_values:
        row_values=[]
        for y in y_values:
            num=len(data[(data[:,1]==y) & (data[:,0]==x)] )
        
            if num==1:
                if data[(data[:,1]==y) & (data[:,0]==x)][0][2]>=CLUSTER:
            
                    row_values.append(-1)
                else:
                    row_values.append(1)
            else:
                row_values.append(num)
        c_values.append(row_values)

#c_values=np.array(c_values)/np.max(c_values)


# Ensure the x_values array has the same length as the number of columns


    cols = len(x_values)
    rows = len(y_values)
# Number of rows in the grid

# Repeat the x_values array to create a 2D array for the grid

    values=np.array(c_values)
# Normalize the values to [0, 1] for color mapping

    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

# Create a colormap
    cmap = plt.get_cmap('bone_r')
    fig, ax = plt.subplots(figsize=(30,8))

# Plot each cell with a color corresponding to the value
    for i in range(rows):
        for j in range(cols):
            if values[i][j]==0:
                color=(0.0,0.0,1.0,0.3)
            elif values[i][j]==1:
                color=(0.0,0.0,0.0,0.01)
                
            elif values[i][j]==-1:
                color=(1.0,0.0,0.0,1.0)
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color,label='outlier')
            else:
                color = cmap(norm(values[i, j]))
            
            if values[i][j]!=-1:
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color)
            ax.add_patch(rect)

    # Set the limits and aspect ratio
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    
    # Remove the axes for better visualization
    
    ax.set_xticks(np.arange(0.5,16.5,1))
    ax.set_xticklabels(np.arange(0,16,1),rotation=90)
    ax.set_yticks(np.arange(0.5,16.5,1))  # Remove y-axis labels
    ax.set_yticklabels(np.arange(0,16,1),rotation=90)
    # Show the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar=plt.colorbar(sm, ax=ax)
    cbar.ax.set_title('Number of data points')
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    
    
    
    # Add legend with unique labels
    plt.legend(unique_labels.values(), unique_labels.keys())
    
    
    # Display the plot
    plt.show()



    