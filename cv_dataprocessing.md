Sign Language Data Processing GuideThis guide outlines the best practices for transforming raw MediaPipe landmark coordinates into a high-performance feature set for Machine Learning models (KNN, Random Forest, SVM, etc.).1. Data Structure OverviewThe input data consists of:Identification: frame_id, timestamp_ms, gesture (Label).Visibility: L_exist, R_exist (Binary flags indicating if hands are in frame).Coordinates: 21 landmarks per hand ($x, y, z$), totaling 126 coordinate features per frame.2. The "Relative Normalization" StrategyRaw coordinates are "absolute" (pixels or 0-1 screen space). This makes the model fail if the user moves to a different part of the camera frame. To fix this, we use Wrist-Relative Normalization.Step 1: Anchor to WristFor every frame where a hand exists:Take the Wrist coordinate (Landmark 0) as the origin $(0,0,0)$.Subtract the Wrist coordinates from all other 20 landmarks.$$X_{rel} = X_{landmark} - X_{wrist}$$$$Y_{rel} = Y_{landmark} - Y_{wrist}$$Step 2: Scale InvarianceTo ensure the model works if the user is close to or far from the camera:Calculate the "Hand Scale" (distance between Landmark 0 and Landmark 9/Middle Finger Base).Divide all relative coordinates by this scale.$$X_{norm} = \frac{X_{rel}}{Scale}$$3. Handling Missing DataIn many frames, only one hand (or no hand) is present.Padding: Do not use 0 for missing hands, as 0 is now a valid relative coordinate (the wrist).Flagging: Use the L_exist and R_exist columns as features so the model "knows" when a hand is missing.4. Feature Engineering: Motion (Velocity)Static hand shapes are often not enough to distinguish signs (e.g., "me" vs. "is"). We must add Temporal Features.Delta Coordinates: For each landmark, calculate the change since the last frame:$$\Delta X = X_t - X_{t-1}$$This allows the model to detect the direction and speed of movement.5. Python ImplementationThe following code implements these improvements on your yes.csv structure.Pythonimport pandas as pd
import numpy as np

def process_gesture_data(file_path):
    df = pd.read_csv(file_path)
    processed_frames = []

    for _, row in df.iterrows():
        features = []
        
        # Process Right Hand (R_x0 to R_z20)
        if row['R_exist'] == 1:
            wrist_x, wrist_y, wrist_z = row['R_x0'], row['R_y0'], row['R_z0']
            
            # Calculate Scale (Distance between Wrist and Middle Finger Base)
            scale = np.sqrt((row['R_x9'] - wrist_x)**2 + (row['R_y9'] - wrist_y)**2)
            
            for i in range(21):
                features.append((row[f'R_x{i}'] - wrist_x) / scale)
                features.append((row[f'R_y{i}'] - wrist_y) / scale)
                features.append((row[f'R_z{i}'] - wrist_z) / scale)
        else:
            # Padding for missing hand
            features.extend([0] * 63)

        # Process Left Hand (L_x0 to L_z20)
        if row['L_exist'] == 1:
            wrist_x, wrist_y, wrist_z = row['L_x0'], row['L_y0'], row['L_z0']
            scale = np.sqrt((row['L_x9'] - wrist_x)**2 + (row['L_y9'] - wrist_y)**2)
            for i in range(21):
                features.append((row[f'L_x{i}'] - wrist_x) / scale)
                features.append((row[f'L_y{i}'] - wrist_y) / scale)
                features.append((row[f'L_z{i}'] - wrist_z) / scale)
        else:
            features.extend([0] * 63)

        processed_frames.append(features)

    return pd.DataFrame(processed_frames)

# Usage
# processed_df = process_gesture_data('yes.csv')