#pragma once
#include "rayHelpers.cu"
#include "perlin.h"
#include "cuda.h"



#define USING_SHADOWS
//#define USING_DOUBLE_TAP_SHADOWS
#define USING_PHOTON_MAPPED_SHADOWS
//#define USING_POINT_LIGHT
//#define STATIC_LIGHT_DIR make_float3(0.0,.71,0.71)
//#define LIGHT_POS make_float3(0,2000,2000)

#define STATIC_LIGHT_DIR make_float3(0.0,1,0)
#define LIGHT_POS make_float3(0,2000,0)
//#define AMBIENT_OCCLUSION
//#define VISUALIZE_BOUNDS



cudaError_t cuda();
__global__ void kernel() {

}


__device__ float currentTime;
__device__ sceneInfo scene;
__device__ unsigned int* lightImage;
__device__ int imageWidth;
__device__ int imageHeight;
//sceneInfo info;


__device__ __forceinline__ void fresnel(const float3& I, const float3& N, const float& ior, float& kr)
{
	float cosi = clamp(-1, 1, dot(I, N));
	float etai = 1, etat = ior;
	if (cosi > 0) { float temp = etai; etai = etat; etat = temp; }// std::swap(etai, etat);
	// Compute sini using Snell's law
	float sint = etai / etat * sqrtf(max(0.f, 1 - cosi * cosi));
	// Total internal reflection
	if (sint >= 1) {
		kr = 1;
	}
	else {
		float cost = sqrtf(max(0.f, 1 - sint * sint));
		cosi = fabsf(cosi);
		float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
		float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
		kr = (Rs * Rs + Rp * Rp) / 2;
	}
	// As a consequence of the conservation of energy, transmittance is given by:
	// kt = 1 - kr;
}


__device__ __forceinline__ float3 refract(const float3& I, const float3& N, const float& ior)
{
	float cosi = clamp(-1, 1, dot(I, N));
	float etai = 1, etat = ior;
	float3 n = N;
	if (cosi < 0) { cosi = -cosi; }
	else { float temp = etai; etai = etat; etat = temp; n = inverse(N); }
	float eta = etai / etat;
	float k = 1 - eta * eta * (1 - cosi * cosi);
	return eta * I + (eta * cosi - sqrtf(k)) * n;
}

__device__ __forceinline__ float3 reflect(const float3& I, const float3& N)
{
	return I - 2 * dot(I, N) * N;
}

__device__ float3 getDistortion(const float3 normal,const float3 inputPos,const int perlinDepth) {

	float d = 0.01;
	float3 rightDir = make_float3(0, 0, 1);
	float3 otherDir1 = cross(rightDir, normal);
	float3 otherDir2 = cross(otherDir1, normal);

	float axis1;
	float axis2;

	if (fabs(normal.x) > fabs(normal.y) && fabs(normal.x) > fabs(normal.z)) {
		axis1 = inputPos.y;
		axis2 = inputPos.z;

	}
	else if (fabs(normal.y) > fabs(normal.z)) {
		axis1 = inputPos.x;
		axis2 = inputPos.z;
	}
	else {
		axis1 = inputPos.x;
		axis2 = inputPos.y;
	}

	

	float sample1 = perlin2d(axis1,  axis2, 1, perlinDepth);
	float sample2 = perlin2d(axis1 + 10000,  axis2 + 100000, 1, perlinDepth);

	float h1 = perlin2d(axis1 - d, axis2, 1, perlinDepth);
	float h2 = perlin2d(axis1 + d, axis2, 1, perlinDepth);
	float h3 = perlin2d(axis1, axis2 - d, 1, perlinDepth);
	float h4 = perlin2d(axis1, axis2 + d, 1, perlinDepth);

	float d1 =  (h2 - h1) / 2 * d;
	float d2 =  (h4 - h3) / 2 * d;

	//float h1 = 


	return (otherDir1 * d1 + otherDir2 * d2);

}

__device__ bool getTranslatedPos(float3 position, float3 &translatedPos) {
	float3 beforeTranslation = make_float3(LIGHT_BUFFER_WORLD_RATIO * position.x, LIGHT_BUFFER_WORLD_RATIO * position.z, LIGHT_BUFFER_THICKNESS_WORLD_RATIO * (position.y +55));
	translatedPos = beforeTranslation + make_float3(0.5, 0.5, 0.5);
	translatedPos = translatedPos * make_float3(LIGHT_BUFFER_WIDTH, LIGHT_BUFFER_WIDTH, LIGHT_BUFFER_THICKNESS);
	return (translatedPos.x >= 0 && translatedPos.x < LIGHT_BUFFER_WIDTH && translatedPos.y >= 0 && translatedPos.y < LIGHT_BUFFER_WIDTH && translatedPos.z >= 0 && translatedPos.z < LIGHT_BUFFER_THICKNESS);
}

__device__ bool worldPositionToLerpedValue(float3 position, float &value) {
	float3 translatedPos;
	bool OK = getTranslatedPos(position, translatedPos);

	if (OK) {
		int currZ = floor(translatedPos.z) * LIGHT_BUFFER_WIDTH * LIGHT_BUFFER_WIDTH;

		int currY = floor(translatedPos.y);
		int currX = floor(translatedPos.x);
		int nextY = min(currY + 1, LIGHT_BUFFER_WIDTH -1);
		int nextX = min(currX + 1, LIGHT_BUFFER_WIDTH -1);

		int outUL = currZ + (nextY * LIGHT_BUFFER_WIDTH + currX);
		int outLL = currZ + (currY * LIGHT_BUFFER_WIDTH + currX);
		int outUR = currZ + (nextY * LIGHT_BUFFER_WIDTH + nextX);
		int outLR = currZ + (currY * LIGHT_BUFFER_WIDTH + nextX);

		float xFactor = translatedPos.x - floor(translatedPos.x);

		//xFactor = pow(xFactor, 2) * 2;
		//xFactor = pow(xFactor, 2) * 2;
		float yFactor = translatedPos.y - floor(translatedPos.y);
		float combinedUpper = lightImage[outUL] * (1.-xFactor) + lightImage[outUR] * (xFactor);
		float combinedDown = lightImage[outLR] * xFactor + lightImage[outLL] * (1. - xFactor);
		float result = combinedUpper* yFactor + (1. - yFactor) * combinedDown;
		value = result;
		return true;
	}
	return false;
}



__device__ bool worldPositionToTextureCoordinate(float3 position, int& out) {
	float3 translatedPos;
	bool res = getTranslatedPos(position, translatedPos);
	out = (((int)(translatedPos.z)) * LIGHT_BUFFER_WIDTH*LIGHT_BUFFER_WIDTH + ((int)translatedPos.y) * LIGHT_BUFFER_WIDTH + (int)(translatedPos.x));
	return res;
}


__device__ hitInfo getHit(const float3 currRayPos,const float3 currRayDir, bool isLightPass) {
	float closestDist = 1000000;
	float3 normal;
	hitInfo toReturn;
	toReturn.hit = false;


	// mathematical objects
	for (int i = 0; i < scene.numObjects; i++) {
		const objectInfo& curr = scene.objects[i];
		float currDist;


		shapeInfo info = curr.shapeData;
		if (info.isMoving) {
			info.pos = make_float3(sin(currentTime*0.3) * info.pos.x, (sin(currentTime * 0.5))*info.pos.y + 2*info.pos.y, cos(currentTime*0.3) * info.pos.z);

		}
		switch (curr.s) {
		case water: {
			shapeInfo otherInfo = info;
			otherInfo.normal = inverse(otherInfo.normal);
			float3 normalToUse = info.normal;
			bool needsToCommunicateInversion = false;
			bool intersected = intersectPlane(info, currRayPos, currRayDir, currDist);
			//if (!intersected) {
			//	intersected = intersectPlane(otherInfo, currRayPos, currRayDir, currDist);
			//	normalToUse = otherInfo.normal;
			//	needsToCommunicateInversion = true;
			//}
			
			if (intersected && currDist < closestDist) {
				closestDist = currDist;
				toReturn.info = curr.rayInfo;
				float3 pos = currRayPos + currDist * currRayDir;
				float3 waveInput = pos * 0.3 + make_float3(1 * currentTime + 10000, 10000, 10000);
				float strength = 3000;

				float3 distortion = getDistortion(normalToUse, waveInput, 4);
				normal = normalize(normalToUse + strength * distortion);
				toReturn.hit = true;
				toReturn.normalIsInversed = needsToCommunicateInversion;

			}

			break;
		}
		case plane: {
			if (intersectPlane(info, currRayPos, currRayDir, currDist) && currDist < closestDist) {
				closestDist = currDist;
				toReturn.info = curr.rayInfo;
				normal = info.normal;
				toReturn.hit = true;
			}

			break;
		}
		case sphere: {
			if (length(info.pos - currRayPos) - info.rad < closestDist && intersectsSphere(currRayPos, currRayDir, info.pos, info.rad, currDist) && currDist < closestDist) {
				closestDist = currDist;
				float3 nextPos = currRayPos + currDist * currRayDir;
				normal = normalize(nextPos - info.pos);
				toReturn.info = curr.rayInfo;
				toReturn.hit = true;

			}
			break;
		}
		}
	}


	// meshes
	for (int i = 0; i < scene.numMeshes; i++) {
		triangleMesh currMesh = scene.meshes[i];

		float tMin = 0;
		float tMax;

		float3 gridPos = (currRayPos - currMesh.bbMin) / currMesh.gridBoxDimensions;
		gridPos = make_float3(floor(gridPos.x), floor(gridPos.y), floor(gridPos.z));

		bool isAlreadyInside = max(gridPos.x, max(gridPos.y, gridPos.z)) < GRID_SIZE && min(gridPos.x, min(gridPos.y, gridPos.z)) >= 0;
		if (isAlreadyInside || (intersectBox(currRayPos, currRayDir, currMesh.bbMin, currMesh.bbMax, tMin, tMax) && tMin < closestDist && tMin > 0)) {

			// engage the GRID
			float3 currPos = currRayPos + (tMin + 0.001)*currRayDir;
			gridPos = (currPos - currMesh.bbMin) / currMesh.gridBoxDimensions;

			int stepsBeforeQuit = 1000;
			bool hitGrid = false;
			while (--stepsBeforeQuit >= 0 && max(gridPos.x, max(gridPos.y, gridPos.z)) < GRID_SIZE && min(gridPos.x, min(gridPos.y, gridPos.z)) >= 0) {

				gridPos = make_float3(floor(gridPos.x), floor(gridPos.y), floor(gridPos.z));
				unsigned int gridPosLoc = GRID_POS(gridPos.x, gridPos.y, gridPos.z);

				float t;
				float u;
				float v;
				for (unsigned int j = 0; j < currMesh.gridSizes[gridPosLoc]; j++) {
					unsigned int iPos = currMesh.grid[gridPosLoc][j];
					if (RayIntersectsTriangle(currRayPos, currRayDir, currMesh.vertices[currMesh.indices[iPos]], currMesh.vertices[currMesh.indices[iPos + 1]], currMesh.vertices[currMesh.indices[iPos + 2]], t, u, v) && t < closestDist) {
						closestDist = t;
						toReturn.info = currMesh.rayInfo;

						normal = (1 - v - u)* currMesh.normals[currMesh.indices[iPos]] + u * currMesh.normals[currMesh.indices[iPos + 1]] + v * currMesh.normals[currMesh.indices[iPos + 2]];
						toReturn.hit = true;
						toReturn.pos = currPos + t * currRayDir;
						stepsBeforeQuit = 1;
					}
				}

				float3 distFromCorner = currPos - gridPos * currMesh.gridBoxDimensions - currMesh.bbMin;
				float3 distFromOtherCorner = currMesh.gridBoxDimensions - distFromCorner;
				float remainingToHitX = max(-distFromCorner.x / currRayDir.x, distFromOtherCorner.x / currRayDir.x);
				float remainingToHitY = max(-distFromCorner.y / currRayDir.y, distFromOtherCorner.y / currRayDir.y);
				float remainingToHitZ = max(-distFromCorner.z / currRayDir.z, distFromOtherCorner.z / currRayDir.z);
				float minDist = min(remainingToHitX, min(remainingToHitY, remainingToHitZ)) + 0.01;

				currPos = currPos + minDist * currRayDir;
				gridPos = (currPos - currMesh.bbMin) / currMesh.gridBoxDimensions;
			}
		}

	}


	toReturn.normal = normal;
	toReturn.pos = currRayPos + closestDist * currRayDir;
	return toReturn;
}



__device__ float getShadowTerm(const float3 originalPos, const float3 normal) {

#ifdef USING_PHOTON_MAPPED_SHADOWS

	float val;
	bool valid = worldPositionToLerpedValue(originalPos, val);
	if (valid) {
		return val*0.01;
	}
	//else {
	//	return 0;
	//}


#endif

#ifndef USING_SHADOWS
	return 1.0;
#endif
	float toReturn;
#ifdef USING_POINT_LIGHT
	float3 toLightVec = normalize(LIGHT_POS - originalPos);
#else 
	float3 toLightVec = STATIC_LIGHT_DIR;
#endif // USING_POINT_LIGHT
	hitInfo hit = getHit(originalPos + 0.001 *toLightVec, toLightVec, false);
#ifdef USING_POINT_LIGHT
	if (!hit.hit || length(hit.pos - originalPos) > length(originalPos - LIGHT_POS)) {
		toReturn = 1.;
	}
	else {
		toReturn = 0.0;
	}
#else 
	if (!hit.hit || length(hit.pos - LIGHT_POS) < 2001.0f) {
		toReturn = 1.;
	}
	else {
		if (hit.info.insideColorDensity > 0.0001) {
			// hack
			toReturn = powf(1. - hit.info.insideColorDensity, length(hit.pos - originalPos));
			//toReturn = (1. - hit.info.insideColorDensity, length(hit.pos - originalPos));
			toReturn = max(0.,toReturn);
			#ifdef USING_DOUBLE_TAP_SHADOWS
			hit = getHit(hit.pos + 0.01 * toLightVec, toLightVec, false);
			toReturn = (!hit.hit || length(hit.pos - LIGHT_POS) < 2001.0f) ? toReturn : 0.;// max(0., toReturn - hit.info.refractivity);
			#endif

			//toReturn = 1;

		}
		else {
			toReturn = 0.0;

		}
	}
#endif // USING_POINT_LIGHT


	return toReturn;

}


struct Ray {
	float3 currRayPos;
	float3 currRayDir;
	hitInfo prevHitToAddDepthFrom;
	float totalContributionRemaining = 0.0;
	bool isLightPass = false;
};

__device__ Ray make_ray(float3 pos, float3 dir, hitInfo prevHit, float remaining, bool lightPass) {
	Ray toReturn;
	toReturn.currRayPos = pos;
	toReturn.currRayDir = dir;
	toReturn.prevHitToAddDepthFrom = prevHit;
	toReturn.totalContributionRemaining = remaining;
	toReturn.isLightPass = lightPass;
	return toReturn;

}

#define MAX_RAYS 5

__device__ float3 traceNonRecursive(const float3 initialRayPos, const float3 initialRayDir, int remainingDepth, const hitInfo prevHitToAddDepthFrom, float totalContributionRemaining = 1.0, bool isLightPass = false) {

	Ray firstRay = make_ray(initialRayPos, initialRayDir, prevHitToAddDepthFrom, totalContributionRemaining, isLightPass);
	float3 accumColor = make_float3(0,0,0);

	int currentNbrRays = 1;
	Ray AllRays[MAX_RAYS];
	AllRays[0] = firstRay;

	for (int i = 0; i < remainingDepth && currentNbrRays > 0; i++) {
		int firstNumR = currentNbrRays;
		for (int j = 0; j < firstNumR; j++) {
			Ray currentRay = AllRays[j];

			hitInfo hit = getHit(currentRay.currRayPos, currentRay.currRayDir, isLightPass);
			if (!hit.hit) {
				accumColor = accumColor + currentRay.totalContributionRemaining * AIR_COLOR;// * currentRay.prevHitToAddDepthFrom.info.color;
			}
			else {
				rayHitInfo info = hit.info;

				float3 reflected = make_float3(0, 0, 0);
				float3 refracted = make_float3(0, 0, 0);
				float3 nextPos = hit.pos;
				float3 normal = hit.normal;

				if (hit.info.roughness > 0.001) {
					float3 distortion = getDistortion(normal, nextPos + make_float3(10000, 10000, 10000), 4);
					normal = normalize(normal + distortion * hit.info.roughness);
				}


				float extraReflection = 0;
				float3 extraColor;
				float3 refractBias = 0.002 * normal;
				float3 reflectBias = refractBias;// 0.001 * normal;
				bool outside = dot(currentRay.currRayDir, normal) < 0;

				//if (currentRay.prevHitToAddDepthFrom.info.insideColorDensity > 0.001) {
				float before = currentRay.totalContributionRemaining;
					float prevColorMP = 1 - powf(1. - currentRay.prevHitToAddDepthFrom.info.insideColorDensity, length(nextPos - currentRay.currRayPos));
					accumColor = accumColor + prevColorMP * currentRay.prevHitToAddDepthFrom.info.color*currentRay.totalContributionRemaining;
					currentRay.totalContributionRemaining *= (1. - prevColorMP);
				//}



				if (info.refractivity* currentRay.totalContributionRemaining > 0.001) {
					float kr = 1.0;
					fresnel(currentRay.currRayDir, normal, outside ? info.refractiveIndex : 1 / info.refractiveIndex, kr);

					if (kr < 1) {
						float3 refractionDirection = normalize(refract(currentRay.currRayDir, normal, info.refractiveIndex));
						float3 refractionRayOrig = outside ? nextPos - refractBias : nextPos + refractBias;

						float refracMP = max(0., (1 - kr));
						//refracted = info.refractivity * refracMP * trace(refractionRayOrig, refractionDirection, remainingDepth - 1, outside ^ hit.normalIsInversed ? hit : hitInfo(), totalContributionRemaining * refracMP, isLightPass);
						Ray nextRay = make_ray(refractionRayOrig, refractionDirection, outside ^ hit.normalIsInversed ? hit : currentRay.prevHitToAddDepthFrom, info.refractivity * refracMP * currentRay.totalContributionRemaining, isLightPass);
						if (currentNbrRays < MAX_RAYS) {
							AllRays[currentNbrRays] = nextRay;
							currentNbrRays++;
						}

					}

					extraReflection = max(0.0, min(1., kr) * info.refractivity);
				}
				float reflecMP = (info.reflectivity + extraReflection)* currentRay.totalContributionRemaining;
				if (reflecMP > 0.001 && !isLightPass) {
					float3 reflectDir = reflect(currentRay.currRayDir, normal);
					float3 reflectionOrig = outside ? nextPos + reflectBias : nextPos - reflectBias;

					Ray nextRay = make_ray(reflectionOrig, reflectDir, currentRay.prevHitToAddDepthFrom, reflecMP, isLightPass);
					if (currentNbrRays < MAX_RAYS) {
						AllRays[currentNbrRays] = nextRay;
						currentNbrRays++;
					}
				}

				float colorMultiplier = max(0., (1. - max(0.f, info.reflectivity) - extraReflection - info.refractivity))* currentRay.totalContributionRemaining;
				float3 color = colorMultiplier * info.color;
				float3 light_dir = STATIC_LIGHT_DIR;
				float angleFactor = (0. + 1.0 * max(0.0, dot(light_dir, normal)));


				if (colorMultiplier > 0.0001 && !isLightPass) {
					float shadowFactor = getShadowTerm(nextPos + 0.01 * inverse(currentRay.currRayDir), normal);
					accumColor = accumColor + ((0.8 * shadowFactor * angleFactor + 0.2) * 1.0 * color) ;
				}
				else if (isLightPass){

					float strength = max(0., (1. - max(0.f, info.reflectivity) - extraReflection - info.refractivity)) * 100 * before;
					float3 translatedPos;
					bool OK = getTranslatedPos(nextPos, translatedPos);
					if (OK) {
						int currZ = ((int)translatedPos.z) * LIGHT_BUFFER_WIDTH * LIGHT_BUFFER_WIDTH;

						int currY = floor(translatedPos.y);
						int currX = floor(translatedPos.x);
						int nextY = min(currY + 1, imageWidth - 1);
						int nextX = min(currX + 1, imageWidth - 1);

						int outUL = currZ + (nextY * LIGHT_BUFFER_WIDTH + currX);
						int outLL = currZ + (currY * LIGHT_BUFFER_WIDTH + currX);
						int outUR = currZ + (nextY * LIGHT_BUFFER_WIDTH + nextX);
						int outLR = currZ + (currY * LIGHT_BUFFER_WIDTH + nextX);

						float xFactor = fmod(translatedPos.x, 1.f);// -floor(translatedPos.x);
						float yFactor = fmod(translatedPos.y, 1.f);

						atomicAdd(&lightImage[outLL], strength * (1. - xFactor) * (1. - yFactor));
						atomicAdd(&lightImage[outUL], strength * (1. - xFactor) * (yFactor));
						atomicAdd(&lightImage[outUR], strength * (xFactor) * (yFactor));
						atomicAdd(&lightImage[outLR], strength * (xFactor) * (1. - yFactor));

					}

				}
			}
			AllRays[j] = AllRays[currentNbrRays - 1];
			currentNbrRays--;
		}
	}
	return accumColor;
}



__device__ float3 trace(const float3 currRayPos, const float3 currRayDir, int remainingDepth, const hitInfo &prevHitToAddDepthFrom, float totalContributionRemaining = 1.0, bool isLightPass = false) {

	hitInfo hit = getHit(currRayPos, currRayDir, isLightPass);

	if (!hit.hit) {
		return AIR_COLOR;
	}
	else {

		rayHitInfo info = hit.info;
		float3 reflected = make_float3(0, 0, 0);
		float3 refracted = make_float3(0, 0, 0);
		float3 nextPos = hit.pos;
		float3 normal = hit.normal;

		if (hit.info.roughness > 0.0001) {
			float3 distortion = getDistortion(normal, nextPos + make_float3(10000,10000,10000), 4);
			normal = normalize(normal + distortion * hit.info.roughness);
		}


		float extraReflection = 0;
		float3 extraColor;
		float3 refractBias = 0.001 * normal;
		float3 reflectBias = 0.0001 * normal;
		float prevColorMP = 0;
		float3 extraPrevColor = make_float3(0,0,0);
		bool outside = dot(currRayDir, normal) < 0;

		if (prevHitToAddDepthFrom.info.insideColorDensity > 0.001) {
			prevColorMP = 1 - powf(1. - prevHitToAddDepthFrom.info.insideColorDensity, length(nextPos - currRayPos)+1);
			extraPrevColor = prevColorMP * prevHitToAddDepthFrom.info.color;
		}

		if (prevColorMP > 0.999 || remainingDepth == 1 || totalContributionRemaining < 0.001)
			return info.color * (1. - prevColorMP) + extraPrevColor;

		if (info.refractivity* totalContributionRemaining > 0.001) {
			float kr = 1.0;
			fresnel(currRayDir, normal, outside ? info.refractiveIndex : 1 / info.refractiveIndex, kr);


			if (kr < 1) {
				float3 refractionDirection = normalize(refract(currRayDir, normal, info.refractiveIndex));
				float3 refractionRayOrig = outside ? nextPos - refractBias : nextPos + refractBias;

				float refracMP = max(0., (1 - kr));
				refracted = info.refractivity * refracMP * trace(refractionRayOrig, refractionDirection, remainingDepth - 1,  outside ^ hit.normalIsInversed ? hit : hitInfo(), totalContributionRemaining* refracMP, isLightPass);
			}
			extraReflection = max(0.0,min(1., kr) * info.refractivity);

		}
		if ((info.reflectivity + extraReflection)* totalContributionRemaining > 0.001 && !isLightPass) {
			float3 reflectDir = reflect(currRayDir, normal);
			float3 reflectionOrig = outside ? nextPos + reflectBias : nextPos - reflectBias;
			float reflecMP = info.reflectivity + extraReflection;

			reflected = reflecMP * trace(reflectionOrig, reflectDir, remainingDepth - 1, prevHitToAddDepthFrom, reflecMP*totalContributionRemaining, isLightPass);
		}

		float colorMultiplier = max(0., (1. - max(0.f,info.reflectivity) - extraReflection - info.refractivity));
			float3 color = colorMultiplier * info.color;
			float3 light_dir = STATIC_LIGHT_DIR;
			float angleFactor = (0. + 1.0 * max(0.0, dot(light_dir, normal)));
			float shadowFactor = 0;
		if (!isLightPass) {
			if (colorMultiplier * (1.-prevColorMP) > 0.1) {
				shadowFactor = getShadowTerm(nextPos + 0.01 * inverse(currRayDir), normal);
			}
			return (1. - prevColorMP) * ((0.8 * shadowFactor * angleFactor + 0.2) * 1.0 * color + reflected + refracted) + extraPrevColor;
		}
		else {

			float strength = (1. - prevColorMP) * colorMultiplier * 100;
			float3 translatedPos;
			bool OK = getTranslatedPos(nextPos, translatedPos);
			if (OK) {
				int currZ = ((int)translatedPos.z) * LIGHT_BUFFER_WIDTH * LIGHT_BUFFER_WIDTH;

				int currY = floor(translatedPos.y);
				int currX = floor(translatedPos.x);
				int nextY = min(currY + 1, imageWidth - 1);
				int nextX = min(currX + 1, imageWidth - 1);

				int outUL = currZ + (nextY * LIGHT_BUFFER_WIDTH + currX);
				int outLL = currZ + (currY * LIGHT_BUFFER_WIDTH + currX);
				int outUR = currZ + (nextY * LIGHT_BUFFER_WIDTH + nextX);
				int outLR = currZ + (currY * LIGHT_BUFFER_WIDTH + nextX);

				float xFactor = fmod(translatedPos.x, 1.f);// -floor(translatedPos.x);
				float yFactor = fmod(translatedPos.y,1.f);

				atomicAdd(&lightImage[outLL], strength*(1. - xFactor) * (1. - yFactor));
				atomicAdd(&lightImage[outUL], strength*(1. - xFactor) * (yFactor));
				atomicAdd(&lightImage[outUR], strength*(xFactor) * (yFactor));
				atomicAdd(&lightImage[outLR], strength*(xFactor) * (1. - yFactor));

			}

			return currRayPos;
		}
	}

}


__global__ void
cudaRender(inputPointers pointers, int imgw, int imgh, float currTime, inputStruct input)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;

	float3 forwardV = make_float3(input.forwardX, input.forwardY, input.forwardZ);
	float3 upV = make_float3(input.upX, input.upY, input.upZ);
	float3 rightV = normalize(cross(upV, forwardV));

	float sizeFarPlane = 10;
	float sizeNearPlane = sizeFarPlane * 0.5;
	float3 origin = make_float3(input.currPosX, input.currPosY, input.currPosZ);
	float distFarPlane = 4;
	float distFirstPlane = distFarPlane * 0.5;

	float3 center = make_float3(imgw / 2.0, imgh / 2.0, 0.);
	float3 distFromCenter = ((x - center.x) / imgw) * rightV + ((center.y - y) / imgh) * upV;
	float3 firstPlanePos = (sizeNearPlane * distFromCenter) + origin + (distFirstPlane * forwardV);
	float3 secondPlanePos = (sizeFarPlane * distFromCenter) + (distFarPlane * forwardV) + origin;

	float3 dirVector = normalize(secondPlanePos - firstPlanePos);


	currentTime = currTime;
	scene = pointers.scene;
	lightImage = pointers.lightImage;
	imageWidth = imgw;
	imageHeight = imgh;
	//float3 out = 255 * 3 * trace(firstPlanePos, dirVector, 10, input.beginMedium, 1.0);
	float3 out = 255 * 3 * traceNonRecursive(firstPlanePos, dirVector, 6, input.beginMedium, 1.0);


	int firstPos = (y * imgw + x) * 4;
	pointers.image1[firstPos] = out.x;
	pointers.image1[firstPos+1] = out.y;
	pointers.image1[firstPos+2] = out.z;
}

//float rand(float2 co) {
//	return fmod((sin(dot(co, make_float2(12.9898, 78.233))) * 43758.5453),0);
//}


__global__ void
cudaLightRender(inputPointers pointers, int imgw, int imgh, float currTime, inputStruct input)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;

	float3 forwardV = STATIC_LIGHT_DIR;// normalize(STATIC_LIGHT_DIR + make_float3(sin(currTime * 0.1), 0, cos(currTime * 0.1)));
	float3 upV = make_float3(1,0,0);
	float3 rightV = normalize(cross(upV, forwardV));
	upV = cross(forwardV, rightV);

	float2 center = make_float2(imgw / 2.0, imgh / 2.0);
	float3 distFromCenter = ((x - center.x) / imgw) * rightV + ((center.y - y) / imgh) * upV;
	float3 startPos = distFromCenter * LIGHT_PLANE_SIZE + forwardV * 400 ;
	float3 dirVector = inverse(forwardV);


	currentTime = currTime;
	scene = pointers.scene;
	lightImage = pointers.lightImage;
	imageWidth = imgw;
	imageHeight = imgh;
	//trace(startPos, dirVector, 10, hitInfo(), 1.0, true);

	traceNonRecursive(startPos, dirVector, 6, input.beginMedium, 1.0, true);


}


__global__ void
cudaClear(unsigned int* buffer, int imgw)
{
	extern __shared__ uchar4 sdata[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int bt = blockDim.z;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;
	int z = blockIdx.z + bt * tz;

	int firstPos = (z * (imgw*imgw) + y * imgw + x);
	buffer[firstPos] = 0;
	//buffer[firstPos + 1] = 0;
	//buffer[firstPos + 2] = 0;
}

extern "C" void
launch_cudaLight(dim3 grid, dim3 block, int sbytes, inputPointers pointers, int imgw, int imgh, float currTime, inputStruct input)
{

	cudaLightRender << < grid, block, sbytes >> > (pointers, imgw, imgh, currTime, input);
}

extern "C" void
launch_cudaClear(dim3 grid, dim3 block, int sbytes, int imgw, unsigned int *buffer)
{

	cudaClear << < grid, block, sbytes >> > (buffer, imgw);
}



extern "C" void
launch_cudaRender(dim3 grid, dim3 block, int sbytes, inputPointers pointers, int imgw, int imgh, float currTime, inputStruct input)
{

	cudaRender << < grid, block, sbytes >> > (pointers, imgw, imgh, currTime, input);
}


