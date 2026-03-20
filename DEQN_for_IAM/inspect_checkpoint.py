import sys
import tensorflow as tf

def inspect(ckpt_path):
    print(f"Inspecting checkpoint: {ckpt_path}")
    try:
        reader = tf.train.load_checkpoint(ckpt_path)
        var_map = reader.get_variable_to_shape_map()
        for key in sorted(var_map.keys()):
            print(f"{key}: {var_map[key]}")
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <path_to_checkpoint_prefix>")
        sys.exit(1)
    inspect(sys.argv[1])
