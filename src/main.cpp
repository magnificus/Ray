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
#include "lodepng/lodepng.h"

#include <iostream>
#include <chrono>
#include <ctime>
#include <vector>

#include "sharedStructs.h"
#include "meshHandler.h"
#include "inputHandler.h"


using namespace std;

// GLFW
GLFWwindow* window;


int num_texels = WIDTH * HEIGHT;
int num_values = num_texels * 4;
int size_tex_data = sizeof(GLuint) * num_values;


int size_light_data = sizeof(GLuint) * LIGHT_BUFFER_WIDTH * LIGHT_BUFFER_WIDTH * LIGHT_BUFFER_THICKNESS;




// OpenGL
GLuint VBO, VAO, EBO;
GLSLShader drawtex_f; // GLSL fragment shader
GLSLShader drawtex_v; // GLSL fragment shader
GLSLProgram shdrawtex; // GLSLS program for textured draw

void* cuda_dev_render_buffer; // stores initial
void* cuda_dev_render_buffer_2; // stores final output
void* cuda_ping_buffer; // used for storing intermediate effects

void* cuda_light_buffer; // result of the light pass
void* cuda_light_buffer_2; // result of the light pass

inputImage waterNormal;


void* cuda_custom_objects_buffer; 
void* cuda_mesh_buffer; 
void* bbm_buffer;



struct cudaGraphicsResource* cuda_tex_resource;
GLuint opengl_tex_cuda;  // OpenGL Texture for cuda result



extern "C" void
launch_cudaBBMRender(dim3 grid, dim3 block, int sbytes, BBMPassInput input);
extern "C" void
launch_cudaRender(dim3 grid, dim3 block, int sbytes, inputPointers pointers, int imgw, int imgh, float currTime, inputStruct input);
extern "C" void
launch_cudaLight(dim3 grid, dim3 block, int sbytes, inputPointers pointers, int imgw, int imgh, float currTime, inputStruct input);
extern "C" void
launch_cudaClear(dim3 grid, dim3 block, int sbytes, int imgw, unsigned int* buffer);
extern "C" void
launch_cudaBloomSample(dim3 grid, dim3 block, int sbytes, int imgw, int imgh, PostProcessPointers pointers);
extern "C" void
launch_cudaBloomOutput(dim3 grid, dim3 block, int sbytes, int imgw, int imgh, PostProcessPointers pointers);
extern "C" void
launch_cudaBlur(dim3 grid, dim3 block, int sbytes, int imgw, int imgh, int currRatio, PostProcessPointers pointers);
extern "C" void
launch_cudaBlur2(dim3 grid, dim3 block, int sbytes, int imgw, int imgh, bool isHorizontal, int currRatio, PostProcessPointers pointers, int dataInterval);
extern "C" void
launch_cudaBlur2SingleChannel(dim3 grid, dim3 block, int sbytes, int imgw, int imgh, bool isHorizontal, int currRatio, PostProcessPointers pointers, int dataInterval);
extern "C" void
launch_cudaDownSampleToHalfRes(dim3 grid, dim3 block, int sbytes, int imgw, int imgh, int currRatio, PostProcessPointers pointers);
extern "C" void
launch_cudaUpSampleToDoubleRes(dim3 grid, dim3 block, int sbytes, int imgw, int imgh, int currRatio, PostProcessPointers pointers);

size_t size_elements_data;

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


void loadImageForCUDA(const char* input_file, unsigned char** cuda_texture, unsigned int &width, unsigned int &height) {
	std::vector<unsigned char> in_image;

	// Load the data
	unsigned error = lodepng::decode(in_image, width, height, input_file, LodePNGColorType::LCT_RGBA);
	if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	std::cout << "texture width: " << width << " height: " << height << std::endl;

	// Prepare the data
	int bufferSize = (in_image.size() * 3) / 4;
	unsigned char* input_image = new unsigned char[bufferSize];
	int where = 0;
	for (int i = 0; i < in_image.size(); ++i) {
		if ((i + 1) % 4 != 0) {
			input_image[where] = in_image.at(i);
			where++;
		}
	}


	checkCudaErrors(cudaMalloc(cuda_texture, bufferSize* sizeof(unsigned char)));
	checkCudaErrors(cudaMemcpy(*cuda_texture, input_image, bufferSize * sizeof(unsigned char), cudaMemcpyHostToDevice));


	delete[] input_image;

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


void createObjects() {
	size_elements_data = sizeof(objectInfo) * NUM_ELEMENTS;
	checkCudaErrors(cudaMalloc(&cuda_custom_objects_buffer, size_elements_data)); // Allocate CUDA memory for objects

	shapeInfo s1 = make_shapeInfo(make_float3(0, -99, 0), make_float3(0, 0, 0), 100);
	shapeInfo s2 = make_shapeInfo(make_float3(0, -15, 50), make_float3(0, 0, 0), 14); // diffuse
	shapeInfo s3 = make_shapeInfo(make_float3(0, 4, -80), make_float3(0, 0, 0), 8); // reflective
	shapeInfo s4 = make_shapeInfo(make_float3(-50, 4, 0), make_float3(0, 0, 0), 12); // refractive
	shapeInfo s5 = make_shapeInfo(make_float3(-50, 10, -50), make_float3(0, 0, 0), 8); // refractive 2
	shapeInfo s6 = make_shapeInfo(make_float3(-30, 10, 10), make_float3(0, 0, 0), 8); // refractive 3
	shapeInfo p1 = make_shapeInfo(make_float3(0, -5, 0), make_float3(0, 1, 0), 0); // water top
	shapeInfo p3 = make_shapeInfo(make_float3(0, -60.0, 0), make_float3(0, 1, 0), 0); // sand bottom

	shapeInfo sun = make_shapeInfo(make_float3(0, 2000, 0), make_float3(1, 0, 0), 200);


	objects[0] = make_objectInfo(sphere, s3, 1.0f, make_float3(0., 1.f, 1.f), 0, 0, 0, 0.0); // reflective
	objects[1] = make_objectInfo(sphere, s6, 0.0f, make_float3(0.0f, 0.0f, 0.1f), 1.0, 1.6, 0.0f, 0.0f); // refractive 3
	objects[2] = make_objectInfo(water, p1, 0.0f, WATER_COLOR, 1.0f, 1.33f, WATER_DENSITY, 1.0f); // water top
	objects[3] = make_objectInfo(plane, p3, 0.f, make_float3(76.0f / 255.0f, 70.0f / 255.f, 50.0f / 255.f), 0, 0, 0.0f, 0); // sand ocean floor
	objects[4] = make_objectInfo(sphere, s1, 0, make_float3(76.0 / 255.0, 70.0 / 255, 50.0 / 255), 0, 0, 0, 0); // island
	objects[5] = make_objectInfo(sphere, sun, 0.0f, 1000 * make_float3(1, 1, 1), 0.0, 1.33, 0.0f, 0.0); // sun
	objects[6] = make_objectInfo(sphere, s2, 0.0f, make_float3(0.5, 0.5, 0), 0.0, 1.4, 0, 1.0f); // yellow boi
	objects[7] = make_objectInfo(sphere, s5, 0.0f, make_float3(0.0, 0.0, 0.1), 1.0, 1.3, 0.0f, 0.0f); // refractive 2
}

void setupLevel() {
	// add objects
	createObjects();

	//float3 sphereCoords = make_float3(1, 0 * 2 * PI / DEFAULT_BBM_SPHERE_RES, 0 * PI / DEFAULT_BBM_SPHERE_RES);
	//sphereCoords = sphericalCoordsToRectangular(sphereCoords);
	//std::cout << "spherical x: " << sphereCoords.x << " y: " << sphereCoords.y << " z: " << sphereCoords.z;

	//float3 simpleman = make_float3(0, 0, -1);
	//float3 sphere = rectangularCoordsToSpherical(simpleman);
	//std::cout << "spherical x: " << sphere.x << " y: " << sphere.y << " z: " << sphere.z;

	//float3 simpleman2 = make_float3(0, -1, 1);
	//float3 sphere2 = rectangularCoordsToSpherical(simpleman2);
	//std::cout << "spherical 2 x: " << sphere2.x << " y: " << sphere2.y << " z: " << sphere2.z;

	//float3 simpleman3 = make_float3(0, 1, 0);
	//float3 sphere3 = rectangularCoordsToSpherical(simpleman3);
	//std::cout << "spherical 3 x: " << sphere3.x << " y: " << sphere3.y << " z: " << sphere3.z;



	// add meshes
	std::vector<triangleMesh> importedMeshes;
	std::vector<rayHitInfo> infos;

	//auto tree = importModel("../../meshes/Palm_Tree.obj", 7.0, make_float3(0, 0, 0), false);
	////auto tree = importModel("../../meshes/leafs.obj", 1.0, make_float3(0, 0, 0), false);
	////importedMeshes.insert(std::end(importedMeshes), std::begin(tree), std::end(tree));
	////infos.push_back(make_rayHitInfo(0.0f, 0.0f, 0.0f, 0.0f, 0.5 * make_float3(133.0 / 255.0, 87.0 / 255.0, 35.0 / 255.0), 0)); // bark
	////infos.push_back(make_rayHitInfo(0.0f, 0.0f, 1.0f, 0.0f, 0.5 * make_float3(111.0 / 255.0, 153.0 / 255, 64.0 / 255), 500)); // palm leaves
	//infos.push_back(make_rayHitInfo(0.0f, 0.0f, 1.0f, 0.0f, 0.7 * make_float3(111.0 / 255.0, 153.0 / 255, 64.0 / 255), 0)); // palm leaves 2

	//std::vector<triangleMesh> rockMesh = importModel("../../meshes/rock.obj", 0.05, make_float3(80.0, -80, 50.0), false);
	//importedMeshes.insert(std::end(importedMeshes), std::begin(rockMesh), std::end(rockMesh));
	//infos.push_back(make_rayHitInfo(0.0f, 0.0f, 1.5f, 0.0f, 0.3 * make_float3(215. / 255, 198. / 255, 171. / 255), 0.f)); //rock

	//std::vector<triangleMesh> bunnyMesh = importModel("../../meshes/bun2.ply", 500, make_float3(0.0, -70, -250.0), false);
	std::vector<triangleMesh> bunnyMesh = importModel("../../meshes/bun2.ply", 500, make_float3(0.0, 0.0, 0.0), false);
	importedMeshes.insert(std::end(importedMeshes), std::begin(bunnyMesh), std::end(bunnyMesh));
	infos.push_back(make_rayHitInfo(0.0, 0.0, 0.0, 0.0, make_float3(20, 0, 0.0), 0)); //le bun

	loadImageForCUDA("../../textures/waternormal.png", &waterNormal.image, waterNormal.width, waterNormal.height);


	size_t size_meshes_data = sizeof(triangleMesh) * importedMeshes.size();
	num_meshes = (unsigned int)importedMeshes.size();

	assert(infos.size() == importedMeshes.size());

	triangleMesh* meshesOnCuda = (triangleMesh*)malloc(size_meshes_data);

	for (int i = 0; i < importedMeshes.size(); i++) {
		triangleMesh curr = importedMeshes[i];
		curr.rayInfo = infos[i];
		meshesOnCuda[i] = prepareMeshForCuda(curr);
	}


	// setup the global grid
	//setupGlobalGrid(objects, importedMeshes);

	checkCudaErrors(cudaMalloc(&cuda_mesh_buffer, size_meshes_data));
	checkCudaErrors(cudaMemcpy(cuda_mesh_buffer, meshesOnCuda, size_meshes_data, cudaMemcpyHostToDevice));


	// ALL BELOW HERE IS WIP
	// just one for now
	size_t size_bbm_data = sizeof(BBMRes) * DEFAULT_BBM_SPHERE_RES * DEFAULT_BBM_SPHERE_RES * DEFAULT_BBM_ANGLE_RES * DEFAULT_BBM_ANGLE_RES;

	BBMRes* BBMTexture;
	checkCudaErrors(cudaMalloc(&BBMTexture, size_bbm_data));


	//triangleMesh* meshesOnCuda2 = (triangleMesh*)malloc(sizeof(triangleMesh));
	importedMeshes[0].rayInfo = infos[0];
	triangleMesh meshOnCuda = prepareMeshForCuda(importedMeshes[0]);

	blackBoxMesh toExec = blackBoxMesh{ BBMTexture, (meshOnCuda.bbMax + meshOnCuda.bbMin) * 0.5, meshOnCuda.rad, DEFAULT_BBM_SPHERE_RES , DEFAULT_BBM_ANGLE_RES };
	BBMPassInput BBMInput = BBMPassInput{ toExec, meshOnCuda };

	dim3 block(8, 8, 1);
	dim3 grid(BBMInput.bbm.sphereResolution / block.x, BBMInput.bbm.sphereResolution / block.y, BBMInput.bbm.angleResolution* BBMInput.bbm.angleResolution);

	launch_cudaBBMRender(grid, block, 0, BBMInput);

	size_t allBBMMeshesSize = sizeof(blackBoxMesh);
	checkCudaErrors(cudaMalloc(&bbm_buffer, allBBMMeshesSize));
	blackBoxMesh* allBBMMeshes = (blackBoxMesh*) malloc(sizeof(blackBoxMesh));
	allBBMMeshes[0] = toExec;
	checkCudaErrors(cudaMemcpy(bbm_buffer, allBBMMeshes, allBBMMeshesSize, cudaMemcpyHostToDevice));
	free(allBBMMeshes);






	//cudaDeviceSynchronize();


	//checkCudaErrors(cudaMemcpy(cuda_mesh_buffer, meshesOnCuda, size_meshes_data, cudaMemcpyHostToDevice));

}

void initCUDABuffers()
{
	// set up vertex data parameters


	cudaError_t stat;
	//size_t myStackSize = 8192;
	//stat = cudaDeviceSetLimit(cudaLimitStackSize, myStackSize);
	checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer, size_tex_data)); // Allocate CUDA memory for color output
	checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer_2, size_tex_data)); // Allocate CUDA memory for color output 2
	checkCudaErrors(cudaMalloc(&cuda_ping_buffer, size_tex_data)); // Allocate CUDA memory for ping buffer
	checkCudaErrors(cudaMalloc(&cuda_light_buffer, size_light_data)); // Allocate CUDA memory for pong buffer
	checkCudaErrors(cudaMalloc(&cuda_light_buffer_2, size_light_data)); // Allocate CUDA memory for pong buffer

	setupLevel();

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

#define MOVE_SPEED 50

void updateObjects(std::chrono::duration<double> deltaTime) {

	if (isMovingObject) {
		float3 movementVec = make_float3(WPressed - SPressed, QPressed - EPressed, APressed - DPressed);
		objects[selectedIndex].shapeData.pos = objects[selectedIndex].shapeData.pos + movementVec * deltaTime.count() * MOVE_SPEED;
	}


	cudaMemcpy(cuda_custom_objects_buffer, objects, size_elements_data, cudaMemcpyHostToDevice);
}

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

	if (!isMovingObject) {

		frontV *= MOVE_SPEED * (WPressed - SPressed) * deltaTime.count();
		rightV *= MOVE_SPEED * (DPressed - APressed) * deltaTime.count();
		lessUp *= MOVE_SPEED * (EPressed - QPressed) * deltaTime.count();
		currP += frontV;
		currP += rightV;
		currP += lessUp;
	}

	input.currPosX = currP.x;
	input.currPosY = currP.y;
	input.currPosZ = currP.z;

	input.forwardX = currFront.x;
	input.forwardY = currFront.y;
	input.forwardZ = currFront.z;

	input.upX = actualUpV.x;
	input.upY = actualUpV.y;
	input.upZ = actualUpV.z;


	updateObjects(deltaTime);

	prevHitInfo prevMedium;
	prevMedium.color = input.currPosY < -5 ? WATER_COLOR : AIR_COLOR;
	prevMedium.insideColorDensity = input.currPosY < -5.f ? WATER_DENSITY : AIR_DENSITY;
	prevMedium.refractiveIndex = input.currPosY < -5.0f ? 1.33f : 1.0f;
	
	input.beginMedium = prevMedium;


	sceneInfo info{ nullptr,nullptr, nullptr, nullptr,(float)totalTime.count(), (objectInfo*)cuda_custom_objects_buffer, NUM_ELEMENTS, (triangleMesh*)cuda_mesh_buffer, (int) num_meshes, (blackBoxMesh*)bbm_buffer, 1 };
	inputPointers pointers{ (unsigned int*)cuda_dev_render_buffer, (unsigned int*)cuda_light_buffer, waterNormal, info };

	// draw light
	dim3 lightGridDraw(LIGHT_BUFFER_WIDTH / block.x, LIGHT_BUFFER_WIDTH / block.y, 1); // 2D grid, every thread will compute a pixel
	dim3 lightGrid(LIGHT_BUFFER_WIDTH / block.x, LIGHT_BUFFER_WIDTH / block.y, LIGHT_BUFFER_THICKNESS / block.z); // 2D grid, every thread will compute a pixel
	launch_cudaClear(lightGrid, block, 0, LIGHT_BUFFER_WIDTH, (unsigned int*)cuda_light_buffer);
	launch_cudaLight(lightGridDraw, block, 0, pointers, LIGHT_BUFFER_WIDTH, LIGHT_BUFFER_WIDTH, (float) totalTime.count(), input);


	if (blurEnabled) {
		launch_cudaBlur2SingleChannel(lightGrid, block, 0, LIGHT_BUFFER_WIDTH, LIGHT_BUFFER_WIDTH, true, 1, PostProcessPointers{ (unsigned int*)cuda_light_buffer, (unsigned int*)cuda_light_buffer, (unsigned int*)cuda_light_buffer_2, (unsigned int*)cuda_dev_render_buffer_2, }, 1); // launch with 0 additional shared memory allocated
		launch_cudaBlur2SingleChannel(lightGrid, block, 0, LIGHT_BUFFER_WIDTH, LIGHT_BUFFER_WIDTH, false, 1, PostProcessPointers{ (unsigned int*)cuda_light_buffer_2, (unsigned int*)cuda_light_buffer_2, (unsigned int*)cuda_light_buffer, (unsigned int*)cuda_dev_render_buffer_2, }, 1); // launch with 0 additional shared memory allocated
	}


	// main render
	launch_cudaRender(grid, block, 0, pointers, WIDTH, HEIGHT, (float)totalTime.count(), input); // launch with 0 additional shared memory allocated


	//// bloom passes
	launch_cudaBloomSample(grid, block, 0, WIDTH, HEIGHT, PostProcessPointers{(unsigned int*)cuda_dev_render_buffer, (unsigned int*)cuda_ping_buffer, (unsigned int*)cuda_ping_buffer, (unsigned int*)cuda_dev_render_buffer_2, }); // launch with 0 additional shared memory allocated
	launch_cudaBlur2(grid, block, 0, WIDTH, HEIGHT, true, 1,PostProcessPointers{(unsigned int*)cuda_dev_render_buffer, (unsigned int*)cuda_ping_buffer, (unsigned int*)cuda_dev_render_buffer_2, (unsigned int*)cuda_dev_render_buffer_2, }, 4); // launch with 0 additional shared memory allocated
	launch_cudaBlur2(grid, block, 0, WIDTH, HEIGHT, false, 1,PostProcessPointers{(unsigned int*)cuda_dev_render_buffer, (unsigned int*)cuda_dev_render_buffer_2, (unsigned int*)cuda_ping_buffer, (unsigned int*)cuda_dev_render_buffer_2, }, 4); // launch with 0 additional shared memory allocated
	launch_cudaBloomOutput(grid, block, 0, WIDTH, HEIGHT, PostProcessPointers{(unsigned int*)cuda_dev_render_buffer, (unsigned int*)cuda_ping_buffer, (unsigned int*)cuda_dev_render_buffer_2, (unsigned int*)cuda_dev_render_buffer_2, }); // launch with 0 additional shared memory allocated

	cudaArray* texture_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));

	checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dev_render_buffer_2, size_tex_data, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));

	cudaDeviceSynchronize();

}

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