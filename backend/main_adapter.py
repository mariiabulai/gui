import sys
import json
import os
import traceback

# 1. Настройка путей
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# Добавляем путь к src, если он есть
if os.path.exists(os.path.join(current_dir, "src")):
    sys.path.append(os.path.join(current_dir, "src"))

# 2. Попытка импорта (Умный поиск)
try:
    # Вариант 1: Если файлы лежат прямо в backend (плоская структура)
    from sfm import SfMReconstructor
except ImportError:
    try:
        # Вариант 2: Если файлы в папке src (структура проекта)
        from src.reconstruction.sfm import SfMReconstructor
    except ImportError:
        try:
             # Вариант 3: Попытка импорта через пакет src
             from reconstruction.sfm import SfMReconstructor
        except ImportError as e:
            print(json.dumps({
                "status": "error", 
                "message": f"CRITICAL: Could not import SfMReconstructor. Check file structure.\nError: {e}\nPath: {sys.path}"
            }))
            sys.exit(1)

def run_reconstruction(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        images_path = config.get('images_path')
        output_path = config.get('output_path')
        calibration_file = config.get('calibration_file')
        
        if not os.path.exists(calibration_file):
            return {"status": "error", "message": f"Calib file missing: {calibration_file}"}

        # Инициализация
        reconstructor = SfMReconstructor(calibration_file)
        
        # Запуск
        result = reconstructor.reconstruct(
            image_dir=images_path,
            output_file=output_path,
            use_ba=True,
            check_quality=False 
        )

        if result:
            return {
                "status": "success", 
                "ply_path": output_path,
                "point_count": len(result[0]) if isinstance(result, tuple) else 0
            }
        else:
            return {"status": "error", "message": "Reconstruction returned empty result"}

    except Exception as e:
        return {"status": "error", "message": f"Python Error: {str(e)}\n{traceback.format_exc()}"}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "No config file passed"}))
        sys.exit(1)
        
    result = run_reconstruction(sys.argv[1])
    print(json.dumps(result))