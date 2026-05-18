from core.data_loader import DataLoader
from core.utils import AppPaths
from core.model_trainer import ModelTrainer
import os

def main():
    AppPaths.load_env()

    DATA_PATH = AppPaths.path(os.getenv("DATA_PATH", "data/processed"))
    MODEL_OUTPUT = AppPaths.path(os.getenv("SIGN_TRANSLATE_MODEL", "models/sign_lstm.keras"))
    ENCODER_OUTPUT = AppPaths.path(os.getenv("LABEL_ENCODER_FILE", "models/sign_lstm_encoder.npy"))

    # Load data
    loader = DataLoader(DATA_PATH, test_size=0.15, val_size=0.15, random_state=42)
    X, y = loader.load_data()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Classes: {loader.get_classes()}")

    # Get splits
    X_train, X_val, X_test, y_train, y_val, y_test = loader.get_splits()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 3. Build and train model
    trainer = ModelTrainer(
        input_shape=(X.shape[1], X.shape[2]),
        num_classes=loader.num_classes,
        config={'learning_rate': 1e-3, 'epochs': 50, 'batch_size': 16},
        model_path=MODEL_OUTPUT
    )
    trainer.build_model()
    trainer.compile_model()
    trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)

    # Save model and encoder
    trainer.save_model(MODEL_OUTPUT, encoder=loader.label_encoder)

if __name__ == "__main__":
    main()