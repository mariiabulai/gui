#include "window.h"
#include "controller.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    
    Window w;
    Controller c(&w); // Контроллер управляет окном
    
    w.show();
    return a.exec();
}