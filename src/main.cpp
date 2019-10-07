// Implementation of CUDA simpleCUDA2GL sample - based on Cuda Samples 9.0
// Dependencies: GLFW, GLEW

#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif

// OpenGL
#include <GL/glew.h> // Take care: GLEW should be included before GLFW
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "libs/helper_cuda.h"
#include "libs/helper_cuda_gl.h"
// C++ libs
#include <string>
#include <filesystem>
#include "shader_tools/GLSLProgram.h"
#include "shader_tools/GLSLShader.h"
#include "gl_tools.h"
#include "glfw_tools.h"


#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


#include <iostream>
#include <chrono>
#include <ctime>

#include "sharedStructs.h"


using namespace std;

// GLFW
GLFWwindow* window;
int WIDTH = 1024;
int HEIGHT = 1024;

int num_texels = WIDTH * HEIGHT;
int num_values = num_texels * 4;
int size_tex_data = sizeof(GLuint) * num_values;


// camera
double currYaw = 270;
double currPitch = 0;
glm::vec3 currFront = glm::vec3(0, 0, -1);

inputStruct input;

// OpenGL
GLuint VBO, VAO, EBO;
GLSLShader drawtex_f; // GLSL fragment shader
GLSLShader drawtex_v; // GLSL fragment shader
GLSLProgram shdrawtex; // GLSLS program for textured draw

// Cuda <-> OpenGl interop resources
void* cuda_dev_render_buffer; // Cuda buffer for initial render
void* cuda_dev_render_buffer_2; // Cuda buffer for initial render


void* cuda_custom_objects_buffer; 
void* cuda_mesh_buffer; 


struct cudaGraphicsResource* cuda_tex_resource;
GLuint opengl_tex_cuda;  // OpenGL Texture for cuda result
extern "C" void


// Forward declaration of CUDA render
launch_cudaRender(dim3 grid, dim3 block, int sbytes, inputPointers pointers, int imgw, int imgh, float currTime, inputStruct input);


extern "C" void
// Forward declaration of CUDA bloom
launch_cudaBloom(dim3 grid, dim3 block, int sbytes, inputPointers pointers, int imgw, int imgh, float currTime, inputStruct input);

// CUDA

size_t size_elements_data;
unsigned int num_elements;

size_t size_meshes_data;
unsigned int num_meshes;

static const char* glsl_drawtex_vertshader_src =
"#version 330 core\n"
"layout (location = 0) in vec3 position;\n"
"layout (location = 1) in vec2 texCoord;\n"
"\n"
"out vec2 ourTexCoord;\n"
"\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(position, 1.0f);\n"
"	ourTexCoord = texCoord;\n"
"}\n";

static const char* glsl_drawtex_fragshader_src =
"#version 330 core\n"
"uniform usampler2D tex;\n"
"in vec2 ourTexCoord;\n"
"out vec4 color;\n"
"void main()\n"
"{\n"
"   	vec4 c = texture(tex, ourTexCoord);\n"
"   	color = c / 255.0;\n"
"}\n";

// QUAD GEOMETRY
GLfloat vertices[] = {
	// Positions             // Texture Coords
	1.0f, 1.0f, 0.5f,1.0f, 1.0f,  // Top Right
	1.0f, -1.0f, 0.5f, 1.0f, 0.0f,  // Bottom Right
	-1.0f, -1.0f, 0.5f, 0.0f, 0.0f,  // Bottom Left
	-1.0f, 1.0f, 0.5f,  0.0f, 1.0f // Top Left 
};
// you can also put positions, colors and coordinates in seperate VBO's
GLuint indices[] = {  
	0, 1, 3,  
	1, 2, 3  
};

// Create 2D OpenGL texture in gl_tex and bind it to CUDA in cuda_tex
void createGLTextureForCUDA(GLuint* gl_tex, cudaGraphicsResource** cuda_tex, unsigned int size_x, unsigned int size_y)
{
	// create an OpenGL texture
	glGenTextures(1, gl_tex); // generate 1 texture
	glBindTexture(GL_TEXTURE_2D, *gl_tex); // set it as current target
	// set basic texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	// Specify 2D texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI_EXT, size_x, size_y, 0, GL_RGB_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
	// Register this texture with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterImage(cuda_tex, *gl_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	SDK_CHECK_ERROR_GL();
}

void initGLBuffers()
{
	// create texture that will receive the result of cuda kernel
	createGLTextureForCUDA(&opengl_tex_cuda, &cuda_tex_resource, WIDTH, HEIGHT);
	// create shader program
	drawtex_v = GLSLShader("Textured draw vertex shader", glsl_drawtex_vertshader_src, GL_VERTEX_SHADER);
	drawtex_f = GLSLShader("Textured draw fragment shader", glsl_drawtex_fragshader_src, GL_FRAGMENT_SHADER);
	shdrawtex = GLSLProgram(&drawtex_v, &drawtex_f);
	shdrawtex.compile();
	SDK_CHECK_ERROR_GL();
}

float WPressed = 0.0;
float SPressed = 0.0;
float DPressed = 0.0;
float APressed = 0.0;
float QPressed = 0.0;
float EPressed = 0.0;

#define PRESSED_MACRO(inKey, variable) if (key == GLFW_KEY_##inKey) { \
if (action == GLFW_PRESS){ \
variable = 1.0; \
} \
else if (action == GLFW_RELEASE) { \
	variable = 0.0; \
} \
}
// Keyboard
void keyboardfunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
	PRESSED_MACRO(W, WPressed);
	PRESSED_MACRO(S, SPressed);
	PRESSED_MACRO(D, DPressed);
	PRESSED_MACRO(A, APressed);
	PRESSED_MACRO(Q, QPressed);
	PRESSED_MACRO(E, EPressed);
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


	currFront.x = cos(glm::radians(currYaw)) * cos(glm::radians(currPitch));
	currFront.y = sin(glm::radians(currPitch));
	currFront.z = sin(glm::radians(currYaw)) * cos(glm::radians(currPitch));
	currFront = glm::normalize(currFront);

}

bool initGL() {
	glewExperimental = GL_TRUE; // need this to enforce core profile
	GLenum err = glewInit();
	glGetError(); // parse first error
	if (err != GLEW_OK) {// Problem: glewInit failed, something is seriously wrong.
		printf("glewInit failed: %s /n", glewGetErrorString(err));
		exit(1);
	}
	glViewport(0, 0, WIDTH, HEIGHT); // viewport for x,y to normalized device coordinates transformation
	SDK_CHECK_ERROR_GL();
	return true;
}


std::vector<triangleMesh> importModel(std::string path, float scale, float3 offset, bool switchYZ = false) {

	std::vector<triangleMesh> toReturn;

	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path, aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_GenSmoothNormals);
	if (!scene) {
		cout << "ya entered an invalid path to mesh fuccboi\n";
		return toReturn;
	}

	for (int i = 0; i < scene->mNumMeshes; i++) {
		auto mesh = scene->mMeshes[i];
		// indices

		triangleMesh current;

		unsigned int totalIndices = 0;

		for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
			auto face = mesh->mFaces[i];
			totalIndices += face.mNumIndices + (face.mNumIndices > 3 ? (face.mNumIndices - 3)*2 : 0);
		}

		current.numIndices = totalIndices;
		current.numVertices = mesh->mNumVertices;
		current.indices = (unsigned int*) malloc(current.numIndices * sizeof(unsigned int));

		unsigned int currIndexPos = 0;
		for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
			auto face = mesh->mFaces[i];
			for (int j = 2; j < face.mNumIndices; j++) { // fan triangulate if not triangles
				current.indices[currIndexPos] = face.mIndices[0];
				current.indices[currIndexPos+1] = face.mIndices[j-1];
				current.indices[currIndexPos+2] = face.mIndices[j];
				currIndexPos += 3;
			}
		}

		// vertices & normals
		current.vertices = (float3*)malloc(current.numVertices * sizeof(float3));
		current.normals = (float3*)malloc(current.numVertices * sizeof(float3));
		cout << "num vertices: " << mesh->mNumVertices << endl;
		cout << "num faces: " << current.numIndices/3 << endl;
		for (unsigned int i = 0; i < current.numVertices; i++) {
			float y = mesh->mVertices[i].y * scale + offset.y;
			float z = mesh->mVertices[i].z * scale + offset.z;
			if (switchYZ) {
				std::swap(y, z);
			}
			current.vertices[i] = make_float3(mesh->mVertices[i].x* scale + offset.x, y, z);
			//cout << "Adding vertex: " << toReturn.vertices[i].x << " " << toReturn.vertices[i].y << " " << toReturn.vertices[i].z << "\n";
			if (mesh->HasNormals())
				current.normals[i] = make_float3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
		}

		toReturn.push_back(current);
	}

	return toReturn;
}

#define MAX(a,b) a > b ? a : b
#define MIN(a,b) a < b ? a : b
#define MOST(type, v1,v2) make_float3( type (v1.x, v2.x), type (v1.y,v2.y), type (v1.z, v2.z));

triangleMesh prepareMeshForCuda(const triangleMesh &myMesh) {
	triangleMesh myMeshOnCuda = myMesh;

	float BIG_VALUE = 1000000;

	float3 max = make_float3(-BIG_VALUE,-BIG_VALUE,-BIG_VALUE);
	float3 min = make_float3(BIG_VALUE, BIG_VALUE, BIG_VALUE);
	for (int i = 0; i < myMesh.numVertices; i++) {
		max = MOST(MAX, max, myMesh.vertices[i]);
		min = MOST(MIN, min, myMesh.vertices[i]);
	}
	

	// acceleration structure
	float3 center = 0.5 * (max + min);
	myMeshOnCuda.bbMax = max;
	myMeshOnCuda.bbMin = min;

	float rad = 0;
	for (int i = 0; i < myMesh.numVertices; i++) {
		rad = MAX(rad, length(myMesh.vertices[i] - center));
	}
	myMeshOnCuda.rad = rad;


	unsigned int gridSize = GRID_SIZE* GRID_SIZE * GRID_SIZE * sizeof(unsigned int*);
	unsigned int gridSizesSize = GRID_SIZE * GRID_SIZE * GRID_SIZE * sizeof(unsigned int);

	unsigned int** grid = (unsigned int**)malloc(gridSize);
	unsigned int* gridSizes = (unsigned int*)malloc(gridSizesSize);


	float3 gridDist = (1.0 / GRID_SIZE) * (max - min);
	myMeshOnCuda.gridBoxDimensions = gridDist;

	for (int x = 0; x < GRID_SIZE; x++) {
		for (int y = 0; y < GRID_SIZE; y++) {
			for (int z = 0; z < GRID_SIZE; z++) {
				std::vector<unsigned int> trianglesToAddToBlock;
				float3 currMin = make_float3(x,y,z)*gridDist + min;
				float3 currMax = make_float3(x+1,y+1,z+1)*gridDist + min;
				float3 currCenter = 0.5 * (currMin + currMax);

				for (int i = 0; i < myMesh.numIndices; i += 3) {
					float3 v0 = myMesh.vertices[myMesh.indices[i]];
					float3 v1 = myMesh.vertices[myMesh.indices[i+1]];
					float3 v2 = myMesh.vertices[myMesh.indices[i+2]];

					float tMin;
					float tMax;
					// we intersect if we're either inside the slab or one edge crosses it
					bool intersecting = (std::fabs(currCenter.x - v0.x) < gridDist.x * 0.5) && (std::fabs(currCenter.y - v0.y) < gridDist.y * 0.5) && (std::fabs(currCenter.z - v0.z) < gridDist.z * 0.5);
					if (!intersecting)
						intersecting |= intersectBox(v0, normalize(v1 - v0), currMin, currMax, tMin, tMax) && tMin > 0 && tMin < length(v1 - v0);
					if (!intersecting)
						intersecting |= intersectBox(v1, normalize(v2 - v1), currMin, currMax, tMin, tMax) && tMin > 0 && tMin < length(v2 - v1);
					if (!intersecting)
						intersecting |= intersectBox(v2, normalize(v0 - v2), currMin, currMax, tMin, tMax) && tMin > 0 && tMin < length(v0 - v2);

					if (intersecting) {
						trianglesToAddToBlock.push_back(i);
					}
				}

				//cout << "x " << x << " y " << y << " z " << z << " collisions: " << trianglesToAddToBlock.size() << endl;
				gridSizes[GRID_POS(x,y,z)] = trianglesToAddToBlock.size();
				grid[GRID_POS(x, y, z)] = (unsigned int*)malloc(trianglesToAddToBlock.size() * sizeof(unsigned int));

				for (int i = 0; i < trianglesToAddToBlock.size(); i++) {
					grid[GRID_POS(x, y, z)][i] = trianglesToAddToBlock[i]; // add collisions to grid
				}
			}
		}
	}

	//checkCudaErrors(cudaMalloc(mesh_pointer, sizeof(triangleMesh)));

	unsigned int indicesSize = myMesh.numIndices * sizeof(unsigned int);
	unsigned int verticesSize = myMesh.numVertices * sizeof(float3);


	if (myMesh.numIndices > 0) {
		// allocate cuda space
		checkCudaErrors(cudaMalloc(&myMeshOnCuda.indices, indicesSize));
		checkCudaErrors(cudaMalloc(&myMeshOnCuda.vertices, verticesSize));
		checkCudaErrors(cudaMalloc(&myMeshOnCuda.normals, verticesSize));
		// this shit is getting convoluted man
		// gotta allocate for each list in grid separately, then feed the correct pointers to the correct positions

		unsigned int** CudaGridPointer = (unsigned int**)malloc(gridSize);

		for (int i = 0; i < GRID_SIZE * GRID_SIZE * GRID_SIZE; i++) {
			checkCudaErrors(cudaMalloc(&(CudaGridPointer[i]), gridSizes[i]*sizeof(unsigned int)));
			checkCudaErrors(cudaMemcpy(CudaGridPointer[i], grid[i], gridSizes[i] * sizeof(unsigned int), cudaMemcpyHostToDevice));

		}
		checkCudaErrors(cudaMalloc(&myMeshOnCuda.grid, gridSize));
		checkCudaErrors(cudaMalloc(&myMeshOnCuda.gridSizes, gridSizesSize));

		// copy data to cuda buffers
		checkCudaErrors(cudaMemcpy(myMeshOnCuda.indices, myMesh.indices, indicesSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(myMeshOnCuda.vertices, myMesh.vertices, verticesSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(myMeshOnCuda.normals, myMesh.normals, verticesSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(myMeshOnCuda.grid, CudaGridPointer, gridSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(myMeshOnCuda.gridSizes, gridSizes, gridSizesSize, cudaMemcpyHostToDevice));



		free(CudaGridPointer);

	}

	for (int i = 0; i < GRID_SIZE * GRID_SIZE * GRID_SIZE; i++) {
		free(grid[i]);
	}
	free(grid);
	free(gridSizes);

	return myMeshOnCuda;
}

void initCUDABuffers()
{
	// set up vertex data parameters


	// We don't want to use cudaMallocManaged here - since we definitely want
	cudaError_t stat;
	size_t myStackSize = 16000;
	stat = cudaDeviceSetLimit(cudaLimitStackSize, myStackSize);
	checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer, size_tex_data)); // Allocate CUDA memory for color output
	checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer_2, size_tex_data)); // Allocate CUDA memory for color output 2


	#define NUM_ELEMENTS 7
	num_elements = NUM_ELEMENTS;
	size_elements_data = sizeof(objectInfo) * num_elements;

	checkCudaErrors(cudaMalloc(&cuda_custom_objects_buffer, size_elements_data)); // Allocate CUDA memory for objects

	shapeInfo s1 = make_shapeInfo(make_float3(0, -99, 0), make_float3(0, 0, 0), 100);
	shapeInfo s2 = make_shapeInfo(make_float3(0, -16, 80), make_float3(0, 0, 0), 8); // diffuse
	shapeInfo s3 = make_shapeInfo(make_float3(0, 4, -40), make_float3(0, 0, 0), 8); // reflective
	shapeInfo s4 = make_shapeInfo(make_float3(-40, 4, 0), make_float3(0, 0, 0), 8); // refractive
	shapeInfo p1 = make_shapeInfo(make_float3(0, -5, 0), make_float3(0, 1, 0), 0); // water top
	shapeInfo p3 = make_shapeInfo(make_float3(0, -60.0, 0), make_float3(0, 1, 0), 0); // sand bottom
	//shapeInfo s4 = make_shapeInfo(make_float3(70, 0, 0), make_float3(1, 0, 0), 0);

	shapeInfo sun = make_shapeInfo(make_float3(1000, 2000, 1000), make_float3(1, 0, 0), 200);


	objectInfo objects[NUM_ELEMENTS];
	//objects[0] = make_objectInfo(sphere, s1, 0.0, make_float3(1, 0, 0), 0, 0, 0);
	objects[0] = make_objectInfo(sphere, s3, 1.0, make_float3(0., 1, 1), 0, 0, 0); // reflective
	objects[1] = make_objectInfo(sphere, s4, 0.0, make_float3(0.0, 0.0, 0.1), 1.0, 1.5, 0.0); // refractive
	objects[2] = make_objectInfo(water, p1, 0.0, make_float3(0,0.0,0.1), 0.9, 1.33, 0.06); // water top
	objects[3] = make_objectInfo(plane, p3, 0, make_float3(76.0 / 255.0, 70.0 / 255, 50.0 / 255), 0, 0, 0.00); // sand ocean floor
	objects[4] = make_objectInfo(sphere, s1, 0.0, make_float3(76.0 / 255.0, 70.0 / 255, 50.0 / 255), 0, 0, 0); // island
	objects[5] = make_objectInfo(sphere, sun, 0.0, 1000*make_float3(1,1,1), 0.0, 1.33, 0.0); // sun
	objects[6] = make_objectInfo(sphere, s2, 0.0, make_float3(0.3, 0.3, 0), 0.0, 1.5, 0); // yellow boi
	//objects[7] = make_objectInfo(plane, p4, 1.0, make_float3(1, 1, 0), 0, 0, 0);

	cudaMemcpy(cuda_custom_objects_buffer, objects, size_elements_data, cudaMemcpyHostToDevice);


	std::vector<triangleMesh> importedMeshes = importModel("C:/Users/Tobbe/Desktop/palmera.obj", 10, make_float3(0, 0, 0), true);
	std::vector<rayHitInfo> infos;
	infos.push_back(make_rayHitInfo( 0.0, 0.0, 0.0, 0.0, 0.5*make_float3(133.0/255.0,87.0/255.0,35.0/255.0))); // bark
	infos.push_back(make_rayHitInfo( 0.0, 0.0, 1.0, 0.1, 0.5*make_float3(111.0/255.0,153.0/255,64.0/255))); // palm leaves

	std::vector<triangleMesh> rockMesh = importModel("C:/Users/Tobbe/Desktop/rock.obj", 0.05, make_float3(50.0, -70, 30.0), false);
	importedMeshes.insert(std::end(importedMeshes), std::begin(rockMesh), std::end(rockMesh));
	infos.push_back(make_rayHitInfo( 0.0, 0.0, 1.5, 0.0, 0.3*make_float3(215./255,198./255,171./255) )); //rock

	std::vector<triangleMesh> bunnyMesh = importModel("C:/Users/Tobbe/Desktop/bun2.ply", 500, make_float3(0.0, -70, 200.0), false);
	importedMeshes.insert(std::end(importedMeshes), std::begin(bunnyMesh), std::end(bunnyMesh));
	infos.push_back(make_rayHitInfo(0.0, 0.0, 1.5, 0.0, make_float3(1,0,0.0))); //le bun

	size_meshes_data = sizeof(triangleMesh) * importedMeshes.size();
	num_meshes = importedMeshes.size();

	assert(infos.size() == importedMeshes.size());

	triangleMesh *meshesOnCuda = (triangleMesh*) malloc(size_meshes_data);

	for (int i = 0; i < importedMeshes.size(); i++) {
		triangleMesh curr = importedMeshes[i];
		curr.rayInfo = infos[i];
		meshesOnCuda[i] = prepareMeshForCuda(curr);
	}

	checkCudaErrors(cudaMalloc(&cuda_mesh_buffer, size_meshes_data));
	checkCudaErrors(cudaMemcpy(cuda_mesh_buffer, meshesOnCuda, size_meshes_data, cudaMemcpyHostToDevice));


}

bool initGLFW() {
	if (!glfwInit()) exit(EXIT_FAILURE);
	// These hints switch the OpenGL profile to core
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	window = glfwCreateWindow(WIDTH, WIDTH, "Raytracer", NULL, NULL);
	if (!window) { glfwTerminate(); exit(EXIT_FAILURE); }
	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);
	glfwSetKeyCallback(window, keyboardfunc);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glfwSetCursorPosCallback(window, mouseFunc);
	return true;
}


#define X_ROTATE_SCALE 0.1
#define Y_ROTATE_SCALE 0.1
#define MOVE_SPEED 50


void generateCUDAImage(std::chrono::duration<double> totalTime, std::chrono::duration<double> deltaTime)
{
	// calculate grid size
	dim3 block(8, 8, 1);
	dim3 grid(WIDTH / block.x, HEIGHT / block.y, 1); // 2D grid, every thread will compute a pixel


	glm::vec3 frontV = currFront;
	glm::vec3 currP(input.currPosX, input.currPosY, input.currPosZ);
	glm::vec3 upV(0, 1, 0);
	glm::vec3 rightV = glm::normalize(glm::cross(frontV, upV));
	glm::vec3 actualUpV = glm::normalize(glm::cross(frontV, rightV));
	glm::vec3 lessUp = actualUpV;

	frontV *= MOVE_SPEED* (WPressed -SPressed)*deltaTime.count();
	rightV *= MOVE_SPEED* (DPressed -APressed)*deltaTime.count();
	lessUp *= MOVE_SPEED * (EPressed - QPressed)*deltaTime.count();
	currP += frontV;
	currP += rightV;
	currP += lessUp;

	input.currPosX = currP.x;
	input.currPosY = currP.y;
	input.currPosZ = currP.z;

	input.forwardX = currFront.x;
	input.forwardY = currFront.y;
	input.forwardZ = currFront.z;

	input.upX = actualUpV.x;
	input.upY = actualUpV.y;
	input.upZ = actualUpV.z;




	sceneInfo info{ totalTime.count(), (objectInfo*)cuda_custom_objects_buffer, num_elements, (triangleMesh*)cuda_mesh_buffer, num_meshes };
	inputPointers pointers{ (unsigned int*)cuda_dev_render_buffer, (unsigned int*)cuda_dev_render_buffer_2, info };


	launch_cudaRender(grid, block, 0, pointers, WIDTH, HEIGHT, totalTime.count(), input); // launch with 0 additional shared memory allocated
	launch_cudaBloom(grid, block, 0, pointers, WIDTH, HEIGHT, totalTime.count(), input); // launch with 0 additional shared memory allocated
	cudaArray* texture_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
	/*checkCudaErrors(*/cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0);



	checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dev_render_buffer_2, size_tex_data, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));

	cudaDeviceSynchronize();


}

//void display(std::chrono::duration<double> duration, std::chrono::duration<double> deltaTime) {
//	generateCUDAImage(duration, deltaTime);
//	glfwPollEvents();
//	// Clear the color buffer
//	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
//	glClear(GL_COLOR_BUFFER_BIT);
//
//	glActiveTexture(GL_TEXTURE0);
//	glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);
//
//	shdrawtex.use(); // we gonna use this compiled GLSL program
//	glUniform1i(glGetUniformLocation(shdrawtex.program, "tex"), 0);
//
//	glBindVertexArray(VAO); // binding VAO automatically binds EBO
//		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
//	glBindVertexArray(0); // unbind VAO
//
//	SDK_CHECK_ERROR_GL();
//	
//	// Swap the screen buffers
//	glfwSwapBuffers(window);
//}


void display(std::chrono::duration<double> duration, std::chrono::duration<double> deltaTime) {
	glClear(GL_COLOR_BUFFER_BIT);
	generateCUDAImage(duration, deltaTime);
	glfwPollEvents();
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	// Swap the screen buffers
	glfwSwapBuffers(window);
}

int main(int argc, char* argv[]) {
	initGLFW();
	initGL();

	printGLFWInfo(window);
	printGlewInfo();
	printGLInfo();

	findCudaGLDevice(argc, (const char**)argv);
	initGLBuffers();
	initCUDABuffers();

	// Generate buffers
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	// Buffer setup
	// Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
	glBindVertexArray(VAO); // all next calls wil use this VAO (descriptor for VBO)

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Position attribute (3 floats)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	// Texture attribute (2 floats)
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound 
	// vertex buffer object so afterwards we can safely unbind
	glBindVertexArray(0);

	// Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs), remember: do NOT unbind the EBO, keep it bound to this VAO
	// A VAO stores the glBindBuffer calls when the target is GL_ELEMENT_ARRAY_BUFFER. 
	// This also means it stores its unbind calls so make sure you don't unbind the element array buffer before unbinding your VAO, otherwise it doesn't have an EBO configured.
	auto firstTime = std::chrono::system_clock::now();
	auto lastTime = firstTime;
	auto lastMeasureTime = firstTime;
	int frameNum = 0;
	// Some computation here


	glBindVertexArray(VAO); // binding VAO automatically binds EBO
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);

	shdrawtex.use(); // we gonna use this compiled GLSL program
	glUniform1i(glGetUniformLocation(shdrawtex.program, "tex"), 0);
	SDK_CHECK_ERROR_GL();


	while (!glfwWindowShouldClose(window))
	{
		auto currTime = std::chrono::system_clock::now();
		auto totalTime = currTime - firstTime;


		display(totalTime, currTime - lastTime);
		std::chrono::duration<double> elapsed_seconds = currTime - lastMeasureTime;
		frameNum++;
		if (elapsed_seconds.count() >= 1.0) {
			// show fps every  second

			std::cout << "fps: " << (frameNum / elapsed_seconds.count()) << "\n";
			frameNum = 0;
			lastMeasureTime = currTime;
		}
		lastTime = currTime;
	}
	glBindVertexArray(0); // unbind VAO


	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}