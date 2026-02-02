#pragma once
#include <QMainWindow>

// Кажемо, що такий клас існує (Forward declaration)
class controller; 

namespace Ui { class window; }

class window : public QMainWindow {
    Q_OBJECT

public:
    // Передаємо вказівник на контролер у конструктор
    explicit window(controller* controller, QWidget *parent = nullptr);
    ~window();

private:
    Ui::window *ui;
    controller *m_controller; // Зберігаємо доступ до мозку
};