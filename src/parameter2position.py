import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

# The `R_matrix_3x3` passed to this function is assumed to be the 3x3 rotation matrix.
# cv2.Rodrigues is typically used to convert a rotation vector to a rotation matrix (or vice versa).
# If your 'R' from camera_parameters['RT'] is already the 3x3 matrix, this function simply returns it.
def get_R_matrix(R_matrix_3x3):
    """
    Returns the 3x3 rotation matrix. Assumes input is already a 3x3 matrix.
    If the input R was a Rodrigues vector, conversion would be needed here.
    """
    return R_matrix_3x3.T

def get_T_world_position(R_matrix_3x3, T_vector_3x1):
    """
    Calculates the camera's optical center position in world coordinates.
    Formula: C_w = -R_transpose * T_camera
    """
    return -np.dot(R_matrix_3x3.T, T_vector_3x1)

def parameter2position(camera_parameters):
    """
    Extracts R matrices and calculates world positions for all cameras.
    Assumes camera_parameters['RT'] contains a list/array of 3x4 [R|T] matrices.
    """
    result = {'R': [], 'T': []}
    for RT_pair in camera_parameters['RT']:
        # Ensure RT_pair is a NumPy array for consistent slicing and operations
        RT_np = np.array(RT_pair)

        # R is the 3x3 rotation matrix (first 3 columns)
        R = RT_np[:, :3]
        # T is the 3x1 translation vector (last column). Reshape to ensure it's a column vector.
        T_vec = RT_np[:, 3].reshape(3, 1)

        # Store the 3x3 rotation matrix directly
        result['R'].append(get_R_matrix(R))

        # Calculate and store the camera's position in world coordinates
        result['T'].append(get_T_world_position(R, T_vec))

    return result

def plot_camera(ax, camera_name, R_matrix_3x3, camera_world_position):
    """
    Plots a single camera's position and its local axes in a 3D plot.
    Args:
        ax (matplotlib.axes.Axes3D): The 3D axes object to plot on.
        camera_name (str): Name of the camera (e.g., 'Camera0').
        R_matrix_3x3 (np.ndarray): The 3x3 rotation matrix for the camera.
        camera_world_position (np.ndarray): The 3x1 vector of the camera's position in world coordinates.
    """
    # Reshape camera position to 1D array for scatter and text plotting
    cam_pos_1d = camera_world_position.reshape(3)

    # Plot camera position
    ax.scatter(*cam_pos_1d, c='k', marker='o', s=50)

    # Plot camera axes (X=right, Y=down, Z=forward)
    # The columns of R_matrix_3x3 are the camera's X, Y, Z axes directions in world coordinates.
    axis_colors = ['red', 'green', 'blue'] # X-axis: red, Y-axis: green, Z-axis: blue
    axis_labels = ['X', 'Y', 'Z'] # Corresponding labels for clarity

    # Adjust the length of the axes for better visualization relative to scene scale
    # You might need to tune this 'axis_length' based on your specific scene's dimensions
    axis_length = 0.5 # Example length, adjust as needed

    for col in range(3):
        # Get the direction vector for the current axis (column of R)
        direction_vector = R_matrix_3x3[:, col]

        # Calculate the end point of the axis line
        end_point = cam_pos_1d + direction_vector * axis_length

        # Plot the axis line from camera position to end_point
        ax.plot(*zip(cam_pos_1d, end_point), color=axis_colors[col], linewidth=2)

        # Add text labels at the end of the axes for clear identification
        ax.text(*end_point, f'{axis_labels[col]}', color=axis_colors[col],
                fontsize=10, ha='center', va='center')
        
    ax.text(
        *cam_pos_1d, camera_name, size=15, color="green", weight="bold"
    )
        
def draw_pyramid_edges(ax: plt.axis, text: str, peak_point: list, rotation: list, size: float = 0.01):
    peak_point_arr = np.squeeze(np.array(peak_point), axis=-1)
    rotation_arr = np.array(rotation)

    # Define vectors from peak to base points
    sizevec = np.array([size * 16, size * 9, -size * 12])
    base_vecs = np.array(
        [
            [-sizevec[0], -sizevec[1], +sizevec[2]],
            [-sizevec[0], +sizevec[1], +sizevec[2]],
            [+sizevec[0], +sizevec[1], +sizevec[2]],
            [+sizevec[0], -sizevec[1], +sizevec[2]],
        ]
    )

    # Rotate vectors and calculate absolute base coordinates
    base_vecs = np.array([np.dot(rotation_arr, bv) for bv in base_vecs])
    base_coords = [peak_point_arr - bv for bv in base_vecs]

    for i, bc in enumerate(base_coords):
        # Draw line from peak to basepoint
        ax.plot3D(
            *zip(peak_point_arr, bc), color="black", linestyle="solid", linewidth=2
        )
        # And to next basepoint
        ax.plot3D(
            *zip(bc, base_coords[(i + 1) % 4]),
            color="black",
            linestyle="solid",
            linewidth=2,
        )

    ax.text(
        *peak_point_arr, text, size=int(size * 1.5 * 1000), color="green", weight="bold"
    )

def show_cameras(camera_positions):
    """
    Creates a 3D plot to visualize all camera positions and orientations.
    """
    fig = plt.figure(figsize=(10, 8)) # Set a good figure size
    ax = fig.add_subplot(projection='3d')

    for camera_idx, (R_matrix, camera_world_pos) in enumerate(zip(camera_positions['R'], camera_positions['T'])):
        plot_camera(ax, f'camera{camera_idx}', R_matrix, camera_world_pos)
        #draw_pyramid_edges(ax, camera_world_pos, R_matrix)
    
    # Set labels for axes
    ax.set_xlabel('X (World)')
    ax.set_ylabel('Y (World)')
    ax.set_zlabel('Z (World)')
    ax.set_title('Camera Positions and Orientations in 3D Space')

    # --- IMPORTANT: Set equal aspect ratio for accurate 3D visualization ---
    # This ensures that distances and angles are not visually distorted.
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    # Calculate the range for each axis
    x_range = abs(xlim[1] - xlim[0])
    y_range = abs(ylim[1] - ylim[0])
    z_range = abs(zlim[1] - zlim[0])

    # Find the maximum range to make all axes equal
    max_range = max([x_range, y_range, z_range])

    # Calculate the new limits for each axis to form a cube
    mid_x = np.mean(xlim)
    mid_y = np.mean(ylim)
    mid_z = np.mean(zlim)

    ax.set_xlim3d([mid_x - max_range / 2, mid_x + max_range / 2])
    ax.set_ylim3d([mid_y - max_range / 2, mid_y + max_range / 2])
    ax.set_zlim3d([mid_z - max_range / 2, mid_z + max_range / 2])

    # Alternatively, for Matplotlib 3.3+
    # ax.set_box_aspect([1,1,1]) # This is a simpler way to force a cubic plot box

    plt.legend(handles=[]) # Display the legend for camera positions
    plt.grid(True) # Show grid for better spatial context
    plt.tight_layout() # Adjust plot to prevent labels from overlapping
    plt.show()

if __name__ == '__main__':
    data_root = './20250818/calibration'

    # Load camera parameters from the pickle file
    camera_parameters = pickle.load(open(os.path.join(data_root, 'camera_parameter.pickle'), 'rb'))

    # Process parameters to get camera positions and R matrices in the desired format
    camera_positions = parameter2position(camera_parameters)

    # Optionally, save the processed camera positions
    SAVE_RESULT = False
    if SAVE_RESULT: pickle.dump(camera_positions, open(os.path.join(data_root, 'camera_position.pickle'), 'wb'))

    # Show the 3D plot of cameras
    show_cameras(camera_positions)
