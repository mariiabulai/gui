#include "Viewer3DWidget.h"
#include <QFile>
#include <QTextStream>
#include <QMouseEvent>
#include <QWheelEvent>
#include <cmath>

Viewer3DWidget::Viewer3DWidget(QWidget *parent) : QOpenGLWidget(parent) {}

void Viewer3DWidget::initializeGL() {
    initializeOpenGLFunctions();
    glClearColor(0.15f, 0.15f, 0.15f, 1.0f); // Темно-серый фон
    glEnable(GL_DEPTH_TEST);
}

void Viewer3DWidget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect = float(w) / (float(h) ? h : 1);
    // Простая перспектива
    float fov = 45.0f;
    float zNear = 0.1f;
    float zFar = 1000.0f;
    float fH = tan(fov / 360 * 3.14159) * zNear;
    float fW = fH * aspect;
    glFrustum(-fW, fW, -fH, fH, zNear, zFar);
    glMatrixMode(GL_MODELVIEW);
}

void Viewer3DWidget::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // Камера
    glTranslatef(0, 0, m_zoom);
    glRotatef(m_rotX, 1, 0, 0);
    glRotatef(m_rotY, 0, 1, 0);

    // Отрисовка точек
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (const auto& p : m_points) {
        glColor3f(p.r / 255.0f, p.g / 255.0f, p.b / 255.0f);
        glVertex3f(p.x, p.y, p.z);
    }
    glEnd();
}

void Viewer3DWidget::loadPly(const QString &filename) {
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) return;

    m_points.clear();
    QTextStream in(&file);
    bool headerEnd = false;

    while (!in.atEnd()) {
        QString line = in.readLine();
        if (line.trimmed() == "end_header") {
            headerEnd = true;
            continue;
        }
        if (!headerEnd) continue;

        QStringList parts = line.split(' ', Qt::SkipEmptyParts);
        if (parts.size() >= 6) {
            Point3D p;
            p.x = parts[0].toFloat();
            p.y = parts[1].toFloat();
            p.z = parts[2].toFloat();
            p.r = parts[3].toFloat();
            p.g = parts[4].toFloat();
            p.b = parts[5].toFloat();
            m_points.push_back(p);
        }
    }
    update();
}

// Тільки ці методи зміни, решту залиш як є

void Viewer3DWidget::mousePressEvent(QMouseEvent *e) {
    // Використовуємо position() для Qt6
    m_lastMousePos = e->position().toPoint();
}

void Viewer3DWidget::mouseMoveEvent(QMouseEvent *e) {
    if (e->buttons() & Qt::LeftButton) {
        // Використовуємо position() для Qt6
        QPoint currentPos = e->position().toPoint();
        int dx = currentPos.x() - m_lastMousePos.x();
        int dy = currentPos.y() - m_lastMousePos.y();
        
        m_rotY += dx * 0.5f;
        m_rotX += dy * 0.5f;
        update();
        m_lastMousePos = currentPos;
    } else {
        // Оновлюємо позицію, навіть якщо кнопка не натиснута (для коректного старту драга)
        m_lastMousePos = e->position().toPoint();
    }
}

void Viewer3DWidget::wheelEvent(QWheelEvent *e) {
    m_zoom += (e->angleDelta().y() / 120.0f);
    update();
}