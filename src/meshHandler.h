#pragma once

#include "sharedStructs.h"
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "libs/helper_cuda.h"
#include "libs/helper_cuda_gl.h"


#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb.h"



std::vector<triangleMesh> importModel(std::string path, float scale, float3 offset, bool switchYZ = false) {

	std::vector<triangleMesh> toReturn;

	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path, aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_GenUVCoords);
	if (!scene) {
		std::cout << "ya entered an invalid path to mesh fuccboi\n";
		return toReturn;
	}

	for (int i = 0; i < scene->mNumMeshes; i++) {
		auto mesh = scene->mMeshes[i];
		// indices

		triangleMesh current;

		unsigned int totalIndices = 0;

		for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
			auto face = mesh->mFaces[i];
			totalIndices += face.mNumIndices + (face.mNumIndices > 3 ? (face.mNumIndices - 3) * 2 : 0);
		}

		current.numIndices = totalIndices;
		current.numVertices = mesh->mNumVertices;
		current.indices = (unsigned int*)malloc(current.numIndices * sizeof(unsigned int));

		unsigned int currIndexPos = 0;
		for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
			auto face = mesh->mFaces[i];
			for (int j = 2; j < face.mNumIndices; j++) { // fan triangulate if not triangles
				current.indices[currIndexPos] = face.mIndices[0];
				current.indices[currIndexPos + 1] = face.mIndices[j - 1];
				current.indices[currIndexPos + 2] = face.mIndices[j];
				currIndexPos += 3;
			}
		}
		

		// vertices & normals
		current.vertices = (float3*)malloc(current.numVertices * sizeof(float3));
		current.normals = (float3*)malloc(current.numVertices * sizeof(float3));
		current.UVs = (float2*)malloc(current.numVertices * sizeof(float2));
		std::cout << "num vertices: " << mesh->mNumVertices << std::endl;
		std::cout << "num faces: " << current.numIndices / 3 << std::endl;
		std::cout << "texture coord channels: " << mesh->GetNumUVChannels() << std::endl;
		for (unsigned int i = 0; i < current.numVertices; i++) {
			float y = mesh->mVertices[i].y * scale + offset.y;
			float z = mesh->mVertices[i].z * scale + offset.z;
			if (switchYZ) {
				std::swap(y, z);
			}
			current.vertices[i] = make_float3(mesh->mVertices[i].x * scale + offset.x, y, z);
			//cout << "Adding vertex: " << toReturn.vertices[i].x << " " << toReturn.vertices[i].y << " " << toReturn.vertices[i].z << "\n";
			if (mesh->HasNormals())
				current.normals[i] = make_float3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
			if (mesh->HasTextureCoords(0))
				current.UVs[i] = make_float2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
		}

		toReturn.push_back(current);
	}

	return toReturn;
}

#define MAX(a,b) a > b ? a : b
#define MIN(a,b) a < b ? a : b
#define MOST(type, v1,v2) make_float3( type (v1.x, v2.x), type (v1.y,v2.y), type (v1.z, v2.z));

triangleMesh prepareMeshForCuda(const triangleMesh& myMesh) {
	triangleMesh myMeshOnCuda = myMesh;

	float BIG_VALUE = 1000000;

	float3 max = make_float3(-BIG_VALUE, -BIG_VALUE, -BIG_VALUE);
	float3 min = make_float3(BIG_VALUE, BIG_VALUE, BIG_VALUE);
	for (int i = 0; i < myMesh.numVertices; i++) {
		max = MOST(MAX, max, myMesh.vertices[i]);
		min = MOST(MIN, min, myMesh.vertices[i]);
	}


	// acceleration structure
	float3 center = 0.5 * (max + min);
	float3 minToCenter = normalize(center - min);
	min = min - minToCenter * 20;
	max = max + minToCenter * 20;

	myMeshOnCuda.bbMax = max;
	myMeshOnCuda.bbMin = min;

	float rad = 0;
	for (int i = 0; i < myMesh.numVertices; i++) {
		rad = MAX(rad, length(myMesh.vertices[i] - center));
	}
	myMeshOnCuda.rad = rad;


	unsigned int gridSize = GRID_SIZE * GRID_SIZE * GRID_SIZE * sizeof(unsigned int*);
	unsigned int gridSizesSize = GRID_SIZE * GRID_SIZE * GRID_SIZE * sizeof(unsigned int);

	unsigned int** grid = (unsigned int**)malloc(gridSize);
	unsigned int* gridSizes = (unsigned int*)malloc(gridSizesSize);


	float3 gridDist = (1.0 / GRID_SIZE) * (max - min);
	myMeshOnCuda.gridBoxDimensions = gridDist;

	for (int x = 0; x < GRID_SIZE; x++) {
		for (int y = 0; y < GRID_SIZE; y++) {
			for (int z = 0; z < GRID_SIZE; z++) {
				std::vector<unsigned int> trianglesToAddToBlock;
				float3 currMin = make_float3(x, y, z) * gridDist + min;
				float3 currMax = make_float3(x + 1, y + 1, z + 1) * gridDist + min;
				float3 currCenter = 0.5 * (currMin + currMax);

				for (int i = 0; i < myMesh.numIndices; i += 3) {
					float3 v0 = myMesh.vertices[myMesh.indices[i]];
					float3 v1 = myMesh.vertices[myMesh.indices[i + 1]];
					float3 v2 = myMesh.vertices[myMesh.indices[i + 2]];

					float tMin;
					float tMax;
					// we intersect if we're either inside the slab or one edge crosses it
					bool intersecting = (std::fabs(currCenter.x - v0.x) < gridDist.x * 0.5) && (std::fabs(currCenter.y - v0.y) < gridDist.y * 0.5) && (std::fabs(currCenter.z - v0.z) < gridDist.z * 0.5);
					intersecting |= intersectBox(v0, normalize(v1 - v0), currMin, currMax, tMin, tMax) && tMin > 0 && tMin < length(v1 - v0);
					intersecting |= intersectBox(v1, normalize(v2 - v1), currMin, currMax, tMin, tMax) && tMin > 0 && tMin < length(v2 - v1);
					intersecting |= intersectBox(v2, normalize(v0 - v2), currMin, currMax, tMin, tMax) && tMin > 0 && tMin < length(v0 - v2);

					if (intersecting) {
						trianglesToAddToBlock.push_back(i);
					}
				}

				//cout << "x " << x << " y " << y << " z " << z << " collisions: " << trianglesToAddToBlock.size() << endl;
				gridSizes[GRID_POS(x, y, z)] = trianglesToAddToBlock.size();
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
	unsigned int UVsSize = myMesh.numVertices * sizeof(float2);


	if (myMesh.numIndices > 0) {
		// allocate cuda space
		checkCudaErrors(cudaMalloc(&myMeshOnCuda.indices, indicesSize));
		checkCudaErrors(cudaMalloc(&myMeshOnCuda.vertices, verticesSize));
		checkCudaErrors(cudaMalloc(&myMeshOnCuda.normals, verticesSize));
		checkCudaErrors(cudaMalloc(&myMeshOnCuda.UVs, UVsSize));
		// this shit is getting convoluted man
		// gotta allocate for each list in grid separately, then feed the correct pointers to the correct positions

		unsigned int** CudaGridPointer = (unsigned int**)malloc(gridSize);

		for (int i = 0; i < GRID_SIZE * GRID_SIZE * GRID_SIZE; i++) {
			checkCudaErrors(cudaMalloc(&(CudaGridPointer[i]), gridSizes[i] * sizeof(unsigned int)));
			checkCudaErrors(cudaMemcpy(CudaGridPointer[i], grid[i], gridSizes[i] * sizeof(unsigned int), cudaMemcpyHostToDevice));

		}
		checkCudaErrors(cudaMalloc(&myMeshOnCuda.grid, gridSize));
		checkCudaErrors(cudaMalloc(&myMeshOnCuda.gridSizes, gridSizesSize));

		// copy data to cuda buffers
		checkCudaErrors(cudaMemcpy(myMeshOnCuda.indices, myMesh.indices, indicesSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(myMeshOnCuda.vertices, myMesh.vertices, verticesSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(myMeshOnCuda.normals, myMesh.normals, verticesSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(myMeshOnCuda.UVs, myMesh.UVs, UVsSize, cudaMemcpyHostToDevice));
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
#define NUM_ELEMENTS 8
objectInfo objects[NUM_ELEMENTS];


//void setupGlobalGrid(objectInfo objects[NUM_ELEMENTS], std::vector<triangleMesh> importedMeshes) {
//
//	unsigned int gridSize = GLOBAL_GRID_SIZE * GLOBAL_GRID_SIZE * GLOBAL_GRID_SIZE * sizeof(unsigned int*);
//	unsigned int gridSizesSize = GLOBAL_GRID_SIZE * GLOBAL_GRID_SIZE * GLOBAL_GRID_SIZE * sizeof(unsigned int);
//
//	float3 gridDist = (1.0 / GLOBAL_GRID_SIZE)* (GLOBAL_GRID_MAX - GLOBAL_GRID_MIN);
//
//
//	unsigned int** objectGrid = (unsigned int**)malloc(gridSize);
//	unsigned int* objectSizes = (unsigned int*)malloc(gridSizesSize);
//	unsigned int** meshesGrid = (unsigned int**)malloc(gridSize);
//	unsigned int* meshesSizes = (unsigned int*)malloc(gridSizesSize);
//
//	std::vector<unsigned int> objectBlocks;
//	for (int x = 0; x < GLOBAL_GRID_SIZE; x++) {
//		for (int y = 0; y < GLOBAL_GRID_SIZE; y++) {
//			for (int z = 0; z < GLOBAL_GRID_SIZE; z++) {
//				std::vector<unsigned int> objectsToAddToBlock;
//				std::vector<unsigned int> meshesToAddToBlock;
//
//				float3 boxMin = GLOBAL_GRID_MIN + GLOBAL_GRID_DIMENSIONS * make_float3(x, y, z);
//				float3 boxMax = GLOBAL_GRID_MIN + GLOBAL_GRID_DIMENSIONS * make_float3(x+1, y+1, z+1);
//				float3 center = (boxMin + boxMax) * 0.5;
//
//				for (int i = 0; i < NUM_ELEMENTS; i++) {
//					objectInfo object = objects[i];
//					switch (object.s) {
//						case water:
//						case plane: {
//							bool foundPositive = false;
//							bool foundNegative = false;
//							for (int x2 = 0; x2 < 2; x2++) {
//								for (int y2 = 0; y2 < 2; y2++) {
//									for (int z2 = 0; z2 < 2; z2++) {
//										float3 vertP = GLOBAL_GRID_MIN + GLOBAL_GRID_DIMENSIONS * make_float3(x2, y2, z2);
//										float3 diffV = vertP - object.shapeData.pos;
//										float dotRes = dot(diffV, object.shapeData.normal);
//										foundPositive |= dotRes > 0;
//										foundNegative |= dotRes < 0;
//
//									}
//								}
//							}
//
//							if (foundNegative && foundPositive)
//								objectsToAddToBlock.push_back(i);
//
//							break;
//						}
//						case sphere: {
//
//						}
//					}
//				}
//				for (int i = 0; i < importedMeshes.size(); i++) {
//					triangleMesh mesh = importedMeshes[i];
//					float3 meshMin = mesh.bbMin;
//					float3 meshMax = mesh.bbMin;
//					#define overlaps1D(var) (meshMax . var >= boxMin .var && boxMax. var >= meshMin. var)
//
//					if (overlaps1D(x) && overlaps1D(y) && overlaps1D(z))
//						objectsToAddToBlock.push_back(i);
//
//				}
//
//				// add objects to grid
//				objectSizes[GLOBAL_GRID_POS(x, y, z)] = objectsToAddToBlock.size();
//				objectGrid[GLOBAL_GRID_POS(x, y, z)] = (unsigned int*)malloc(objectsToAddToBlock.size() * sizeof(unsigned int));
//
//				for (int i = 0; i < objectsToAddToBlock.size(); i++) {
//					objectGrid[GLOBAL_GRID_POS(x, y, z)][i] = objectsToAddToBlock[i]; // add collisions to grid
//				}
//
//				// add meshes to grid
//				meshesSizes[GLOBAL_GRID_POS(x, y, z)] = meshesToAddToBlock.size();
//				meshesGrid[GLOBAL_GRID_POS(x, y, z)] = (unsigned int*)malloc(meshesToAddToBlock.size() * sizeof(unsigned int));
//
//				for (int i = 0; i < meshesToAddToBlock.size(); i++) {
//					meshesGrid[GLOBAL_GRID_POS(x, y, z)][i] = meshesToAddToBlock[i]; // add collisions to grid
//				}
//			}
//		}
//	}
//
//	//unsigned int indicesSize = myMesh.numIndices * sizeof(unsigned int);
//	//unsigned int verticesSize = myMesh.numVertices * sizeof(float3);
//
//	//unsigned int 
//
//	//void** gridMeshes;
//	//void** gridObjects;
//	//void* gridMeshesSizes;
//	//void* gridObjectsSizes;
//
//
//	checkCudaErrors(cudaMalloc(&gridMeshes, gridSize));
//	checkCudaErrors(cudaMalloc(&gridObjects, gridSize));
//
//	checkCudaErrors(cudaMalloc(&gridMeshesSizes, gridSizesSize));
//	checkCudaErrors(cudaMalloc(&gridObjectsSizes, gridSizesSize));
//
//	unsigned int** CudaMeshGridPointer = (unsigned int**)malloc(gridSize);
//	unsigned int** CudaObjectGridPointer = (unsigned int**)malloc(gridSize);
//	for (int i = 0; i < GLOBAL_GRID_SIZE * GLOBAL_GRID_SIZE * GLOBAL_GRID_SIZE; i++) {
//			checkCudaErrors(cudaMalloc(&(CudaMeshGridPointer[i]), meshesSizes[i] * sizeof(unsigned int)));
//			checkCudaErrors(cudaMemcpy(CudaMeshGridPointer[i], meshesGrid[i], meshesSizes[i] * sizeof(unsigned int), cudaMemcpyHostToDevice));
//
//			checkCudaErrors(cudaMalloc(&(CudaObjectGridPointer[i]), objectSizes[i] * sizeof(unsigned int)));
//			checkCudaErrors(cudaMemcpy(CudaObjectGridPointer[i], objectGrid[i], objectSizes[i] * sizeof(unsigned int), cudaMemcpyHostToDevice));
//	}
//
//	checkCudaErrors(cudaMalloc(&gridMeshes, gridSize));
//	checkCudaErrors(cudaMalloc(&gridObjects, gridSize));
//	checkCudaErrors(cudaMalloc(&gridMeshesSizes, gridSizesSize));
//	checkCudaErrors(cudaMalloc(&gridObjectsSizes, gridSizesSize));
//
//
//	checkCudaErrors(cudaMemcpy(gridMeshes, CudaMeshGridPointer, gridSize, cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(gridMeshesSizes, gridMeshesSizes, gridSizesSize, cudaMemcpyHostToDevice));
//
//	checkCudaErrors(cudaMemcpy(gridObjects, CudaObjectGridPointer, gridSize, cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(gridObjectsSizes, gridObjectsSizes, gridSizesSize, cudaMemcpyHostToDevice));
//
//
//	for (int i = 0; i < GLOBAL_GRID_SIZE * GLOBAL_GRID_SIZE * GLOBAL_GRID_SIZE; i++) {
//		free(meshesGrid[i]);
//		free(objectGrid[i]);
//	}
//	free(meshesGrid);
//	free(objectGrid);
//	free(meshesSizes);
//	free(objectSizes);
//
//	//return myMeshOnCuda;
//
//
//}
