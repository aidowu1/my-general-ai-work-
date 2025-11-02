# python/predict_py.py
import numpy as np
import onnxruntime as ort
import os
import sys
import csv

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.onnx")

def load_samples_from_csv(path):
    samples = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row: continue
            # parse floats, ignore empty cells
            values = [float(x) for x in row if x.strip()!='']
            if len(values)!=4:
                raise ValueError(f"Expected 4 features per row, got {len(values)}: {row}")
            samples.append(values)
    return np.array(samples, dtype=np.float32)

def predict(samples):
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print(f"Loaded model from {MODEL_PATH}")
    input_name = sess.get_inputs()[0].name
    print(f"Input name: {input_name}, shape: {sess.get_inputs()[0].shape}, type: {sess.get_inputs()[0].type}")
    output_names = [output.name for output in sess.get_outputs()]
    print(f"Output names: {output_names}")
    
    outputs = sess.run(None, {input_name: samples})
    return outputs

if __name__ == "__main__":
    if len(sys.argv)>1:
        csv_path = sys.argv[1]
        samples = load_samples_from_csv(csv_path)
        print(f"Loaded {len(samples)} samples from {csv_path}")
    else:
        samples = np.array([[5.1,3.5,1.4,0.2]], dtype=np.float32)
        print("Using default single sample:", samples)

    outputs = predict(samples)
    print("Raw outputs:", outputs)
    # print label indices if present
    if len(outputs)>0:
        labels = outputs[0]
        print("Predicted label indices:", labels)
