cmake_minimum_required(VERSION 3.16)

project(SfM_GUI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# === ВАЖНО: УКАЖИ ПУТЬ К QT ЗДЕСЬ ===
# Проверь версию папки в D:/Qt/ (например, 6.5.3 или 6.6.1) и впиши сюда:
set(CMAKE_PREFIX_PATH "D:/Qt/6.8.3/mingw_64") 
# ====================================

find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGLWidgets Gui Core)

set(SOURCES
    main.cpp
    window.cpp window.h
    controller.cpp controller.h
    PythonBridge.cpp PythonBridge.h
    Viewer3DWidget.cpp Viewer3DWidget.h
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/window.ui")
    list(APPEND SOURCES window.ui)
endif()

add_executable(SfM_GUI ${SOURCES})

target_link_libraries(SfM_GUI PRIVATE
    Qt6::Widgets
    Qt6::OpenGLWidgets
    Qt6::Gui
    Qt6::Core
    opengl32
)