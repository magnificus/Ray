#pragma once

#define X_ROTATE_SCALE 0.1
#define Y_ROTATE_SCALE 0.1


// camera
double currYaw = 270;
double currPitch = 0;
glm::vec3 currFront = glm::vec3(0, 0, -1);

inputStruct input;



float WPressed = 0.0;
float SPressed = 0.0;
float DPressed = 0.0;
float APressed = 0.0;
float QPressed = 0.0;
float EPressed = 0.0;

bool isMovingObject = false;
int selectedIndex;

float Pressed1 = 0.0;
float Pressed2 = 0.0;
float Pressed3 = 0.0;
float Pressed4 = 0.0;
float Pressed5 = 0.0;
float Pressed6 = 0.0;

bool blurEnabled = false;

#define PRESSED_RELEASED_MACRO(inKey, variable) if (key == GLFW_KEY_##inKey) { \
if (action == GLFW_PRESS){ \
variable = 1; \
} \
else if (action == GLFW_RELEASE) { \
	variable = 0; \
} \
}

#define PRESSED_ONLY_MACRO(inKey, variable, val) if (key == GLFW_KEY_##inKey) { \
if (action == GLFW_PRESS){ \
variable = val; \
} \
}

// Keyboard
void keyboardfunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
	PRESSED_RELEASED_MACRO(W, WPressed);
	PRESSED_RELEASED_MACRO(S, SPressed);
	PRESSED_RELEASED_MACRO(D, DPressed);
	PRESSED_RELEASED_MACRO(A, APressed);
	PRESSED_RELEASED_MACRO(Q, QPressed);
	PRESSED_RELEASED_MACRO(E, EPressed);

	PRESSED_ONLY_MACRO(SPACE, isMovingObject, !isMovingObject);
	PRESSED_ONLY_MACRO(P, blurEnabled, !blurEnabled);
	PRESSED_ONLY_MACRO(0, selectedIndex, 0);
	PRESSED_ONLY_MACRO(1, selectedIndex, 1);
	PRESSED_ONLY_MACRO(2, selectedIndex, 2);
	PRESSED_ONLY_MACRO(3, selectedIndex, 3);
	PRESSED_ONLY_MACRO(4, selectedIndex, 4);
	PRESSED_ONLY_MACRO(5, selectedIndex, 5);
	PRESSED_ONLY_MACRO(6, selectedIndex, 6);
	PRESSED_ONLY_MACRO(7, selectedIndex, 7);
}

bool firstMouse = true;
double mouseDeltaX;
double mouseDeltaY;

double lastX;
double lastY;

void mouseFunc(GLFWwindow* window, double xpos, double ypos) {

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;
	lastX = xpos;
	lastY = ypos;

	float sensitivity = 0.05;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	currYaw += xoffset;
	currPitch += yoffset;

	currPitch = currPitch > 89.0f ? 89.0f : currPitch;
	currPitch = currPitch < -89.0f ? -89.0f : currPitch;


	currFront.x = (float)cos(glm::radians(currYaw)) * cos(glm::radians(currPitch));
	currFront.y = (float)sin(glm::radians(currPitch));
	currFront.z = (float)sin(glm::radians(currYaw)) * cos(glm::radians(currPitch));
	currFront = glm::normalize(currFront);

}