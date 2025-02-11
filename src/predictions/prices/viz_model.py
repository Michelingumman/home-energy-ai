import visualkeras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from pathlib import Path
import PIL
from PIL import ImageFont
import logging
import tensorflow as tf
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def launch_tensorboard():
    """Launch TensorBoard server"""
    import webbrowser
    from tensorboard import program
    
    # Remove all previous TensorBoard instances
    tf.keras.backend.clear_session()
    
    # Launch TensorBoard
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'models/logs'])
    url = tb.launch()
    print(f"\nTensorBoard started at {url}")
    print("Opening in default browser...")
    webbrowser.open(url)
    return tb

class ModelVisualizer:
    def __init__(self):
        # Setup paths
        self.project_root = Path(__file__).parents[3]
        self.models_dir = self.project_root / "models/saved"
        self.plots_dir = self.project_root / "models/saved"
        self.logs_dir = self.project_root / "models/logs"
        
        # Create directories if they don't exist
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def create_model(self):
        """Create the model architecture"""
        model = Sequential([
            LSTM(64, input_shape=(48, 9), return_sequences=True, name="LSTM_1"),
            Dropout(0.2, name="Dropout_1"),
            LSTM(32, name="LSTM_2"),
            Dropout(0.2, name="Dropout_2"),
            Dense(16, activation='relu', name="Dense_1"),
            Dense(1, name="Output")
        ])
        return model
    
    def setup_tensorboard(self, model):
        """Setup TensorBoard callback and write model graph"""
        log_dir = self.logs_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create TensorBoard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        
        # Write the model graph
        writer = tf.summary.create_file_writer(str(log_dir))
        with writer.as_default():
            tf.summary.graph(tf.function(lambda x: model(x)).get_concrete_function(
                tf.TensorSpec(shape=(None, 48, 9), dtype=tf.float32)).graph)
        
        logging.info(f"TensorBoard logs directory: {log_dir}")
        logging.info("To view TensorBoard, run: tensorboard --logdir models/logs")
        
        return tensorboard_callback
    
    def create_dot_visualization(self, model):
        """Create dot model visualization using graphviz"""
        try:
            dot_file = self.plots_dir / "model_architecture_dot.png"
            plot_model(
                model,
                to_file=str(dot_file),
                show_shapes=True,
                show_layer_names=True,
                show_dtype=True,
                rankdir="TB",
                expand_nested=True,
                dpi=200,
                layer_range=None,
                show_layer_activations=True
            )
            logging.info(f"Saved dot visualization to {dot_file}")
        except Exception as e:
            logging.error(f"Failed to create dot visualization: {str(e)}")
    
    def visualize_model(self, model=None, save=True, launch_tb=True):
        """Create visualization of the model architecture"""
        if model is None:
            try:
                # Try to load saved model
                model = load_model(self.models_dir / "best_model.keras")
                logging.info("Using saved model for visualization")
            except:
                # Create new model if loading fails
                model = self.create_model()
                logging.info("Using new model for visualization")
        
        # Color scheme with proper format
        color_map = {
            LSTM: {'fill': '#2ecc71'},     # Green for LSTM layers
            Dropout: {'fill': '#e74c3c'},  # Red for Dropout layers
            Dense: {'fill': '#3498db'}     # Blue for Dense layers
        }
        
        # Try to load a font (fallback to default if not found)
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = None
            logging.warning("Could not load Arial font, using default")
        
        # Create layered visualization
        layered = visualkeras.layered_view(
            model,
            legend=True,
            color_map=color_map,
            to_file=str(self.plots_dir / "model_architecture_layered.png") if save else None,
            spacing=50,
            draw_volume=True,
            scale_z=1.5
        )
        
        # Create graph visualization with simpler settings
        graph = visualkeras.graph_view(
            model,
            to_file=str(self.plots_dir / "model_architecture_graph.png") if save else None,
            show_shapes=True,
            show_layer_names=True
        )
        
        # Create dot visualization
        self.create_dot_visualization(model)
        
        # Setup TensorBoard
        tensorboard_callback = self.setup_tensorboard(model)
        
        if launch_tb:
            try:
                tb = launch_tensorboard()
                print("\nPress Ctrl+C to stop TensorBoard server when done")
            except Exception as e:
                logging.error(f"Failed to launch TensorBoard: {str(e)}")
                print("\nTo manually start TensorBoard, run:")
                print("python -m tensorboard --logdir models/logs")
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        if save:
            logging.info(f"Saved visualizations to {self.plots_dir}")
        
        return layered, graph, tensorboard_callback

def main():
    visualizer = ModelVisualizer()
    
    # Create and visualize model
    model = visualizer.create_model()
    visualizer.visualize_model(model, launch_tb=True)
    
    # Print architecture details
    print("\nDetailed Architecture:")
    print("Input Shape: (48 hours, 9 features)")
    print("\nFeatures:")
    print("1. Price")
    print("2-3. Hour (sin, cos)")
    print("4-5. Day of Week (sin, cos)")
    print("6-7. Month (sin, cos)")
    print("8. Peak Hour Flag")
    print("9. Weekend Flag")
    
    print("\nVisualization files created:")
    print("1. model_architecture_layered.png - 3D layer visualization")
    print("2. model_architecture_graph.png - Network graph visualization")
    print("3. model_architecture_dot.png - Detailed dot visualization")
    print("\nTensorBoard logs created in models/logs directory")
    
    print("\nNote: TensorBoard server is running in background.")
    print("Press Ctrl+C to stop the server and exit.")
    
    try:
        # Keep the script running for TensorBoard
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        print("\nStopping TensorBoard server...")
    finally:
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()
