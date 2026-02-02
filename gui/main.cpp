#include <QApplication>
#include "window.h"
#include "controller.h"
// Включення заглушки Viewer3DWidget
#include "Viewer3DWidget.h" 

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    a.setStyleSheet(
        // Загальний фон вікна - темно-сірий
        "QMainWindow { background-color: #ECE3D2; }"
        
        // Всі тексти - світлі, шрифт трохи більший
        "QWidget { color: #1F1F1F; font-family: 'Segoe UI', sans-serif; font-size: 14px; }"
        
        // Списки (де фото) - трохи світліший сірий
        "QListWidget { background-color: #ece4d7ff; border: 1px solid #aeaeaeff; border-radius: 6px; padding: 5px; }"
        
        // Кнопки - сині, з заокругленими кутами
        "QPushButton { background-color: #9FAA74; color: white; border-radius: 6px; padding: 8px 16px; font-weight: bold; }"
        "QPushButton:hover { background-color: #D7DAB3; }"  // Колір при наведенні
        "QPushButton:pressed { background-color: #7a8453ff; }" // Колір при натисканні
        
        // Прогрес бар
        "QProgressBar { border: 2px solid #555; border-radius: 5px; text-align: center; color: white; }"
        "QProgressBar::chunk { background-color: #4A6644; width: 20px; }" // Зелена лінія завантаження
        "QListWidget { "
        "   background-color: #cacaca; "    // Сірий фон
        "   border: 1px solid #aeaeae; "
        "   border-radius: 6px; "
        "   padding: 5px; "
        "   color: #1F1F1F;"                // Колір тексту файлів
        "}"
        
        // Колір елемента, коли на нього навели мишкою (hover)
        "QListWidget::item:hover { "
        "   background-color: #e0e0e0; "
        "}"

        // Колір елемента, коли його вибрали (selected) - замість синього робимо зеленим
        "QListWidget::item:selected { "
        "   background-color: #9FAA74; "    // Той самий зелений, що у кнопок
        "   color: white; "
        "}"
    );
    
    // 1. Створюємо Controller (Модуль 4)
    controller controller; 
    
    // 2. Створюємо GUI (Модуль 1), передаючи Controller
    window w(&controller);
    
    // Примітка: В Qt Designer вам потрібно буде додати Viewer3DWidget 
    // або додати його вручну, щоб MainWindow міг викликати його метод.
    
    w.show();
    
    return a.exec();
}