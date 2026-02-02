cmake_minimum_required(VERSION 3.16)

# Назва вашого проекту
project(MyReconstructionApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Знаходимо Qt6 на комп'ютері
set(CMAKE_PREFIX_PATH "D:/Qt/6.8.3/mingw_64")

find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGLWidgets)

# Вказуємо, що це Qt-проект (автоматично обробляє moc-файли)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Додаємо всі ваші .cpp та .h файли
add_executable(MyReconstructionApp
    main.cpp
    window.cpp
    window.h
    controller.cpp
    controller.h
    PythonBridge.cpp  # Ваша заглушка
    PythonBridge.h
    Viewer3DWidget.h  # Ваша заглушка
    # Якщо є ui-файл від дизайнера, додайте його сюди:
    window.ui 
)

# Лінкуємо бібліотеки Qt до проекту
target_link_libraries(MyReconstructionApp PRIVATE Qt6::Widgets Qt6::OpenGLWidgets)