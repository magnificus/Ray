#pragma once
#include "rayHelpers.cu"
#include "perlin.h"



#define USING_SHADOWS
//#define USING_DOUBLE_TAP_SHADOWS
#define USING_PHOTON_MAPPED_SHADOWS
//#define LIGHT_PASS_USES_REFLECTION
//#define USING_POINT_LIGHT
//#define STATIC_LIGHT_DIR make_float3(0.0,.71,0.71)
//#define LIGHT_POS make_float3(0,2000,2000)

#define STATIC_LIGHT_DIR make_float3(0.0,1,0)
#define LIGHT_POS make_float3(0,2000,0)
//#define AMBIENT_OCCLUSION
//#define VISUALIZE_BOUNDS


#define MAX_DISTANCE_FROM_CAMERA_FOR_EFFECTS 1000



cudaError_t cuda();
__global__ void kernel() {

}


__device__ float currentTime;

__device__ inputPointers iPointers = inputPointers{ nullptr, 0,0 };
__device__ int imageWidth;
__device__ int imageHeight;
__device__ float3 startPos;
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

__device__ float3 getDistortion(const float3 normal, const float3 inputPos, const int perlinDepth) {

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



	float sample1 = perlin2d(axis1, axis2, 1, perlinDepth);
	float sample2 = perlin2d(axis1 + 10000, axis2 + 100000, 1, perlinDepth);

	float h1 = perlin2d(axis1 - d, axis2, 1, perlinDepth);
	float h2 = perlin2d(axis1 + d, axis2, 1, perlinDepth);
	float h3 = perlin2d(axis1, axis2 - d, 1, perlinDepth);
	float h4 = perlin2d(axis1, axis2 + d, 1, perlinDepth);

	float d1 = (h2 - h1) / 2 * d;
	float d2 = (h4 - h3) / 2 * d;


	return (otherDir1 * d1 + otherDir2 * d2);

}

__device__ bool getTranslatedPos(float3 position, float3& translatedPos) {
	float3 beforeTranslation = make_float3(LIGHT_BUFFER_WORLD_RATIO * position.x, LIGHT_BUFFER_WORLD_RATIO * position.z, LIGHT_BUFFER_THICKNESS_WORLD_RATIO * position.y);
	translatedPos = beforeTranslation + make_float3(0.5, 0.5, 0.5);
	translatedPos = translatedPos * make_float3(LIGHT_BUFFER_WIDTH, LIGHT_BUFFER_WIDTH, LIGHT_BUFFER_THICKNESS);
	return max(max(abs(beforeTranslation.x), abs(beforeTranslation.y)), abs(beforeTranslation.z)) <= 0.5;
}

__device__ bool worldPositionToLerpedValue(float3 position, float& value) {
	float3 translatedPos;
	bool OK = getTranslatedPos(position, translatedPos);

	if (OK) {
		int currY = floor(translatedPos.y);
		int currX = floor(translatedPos.x);
		int nextY = min(currY + 1, imageWidth - 1);
		int nextX = min(currX + 1, imageWidth - 1);
		int currZ = floor(translatedPos.z) * LIGHT_BUFFER_WIDTH * LIGHT_BUFFER_WIDTH;
		int nextZ = min(LIGHT_BUFFER_THICKNESS - 1., floor(translatedPos.z + 1)) * LIGHT_BUFFER_WIDTH * LIGHT_BUFFER_WIDTH;

		int outDUL = currZ + (nextY * LIGHT_BUFFER_WIDTH + currX);
		int outDLL = currZ + (currY * LIGHT_BUFFER_WIDTH + currX);
		int outDUR = currZ + (nextY * LIGHT_BUFFER_WIDTH + nextX);
		int outDLR = currZ + (currY * LIGHT_BUFFER_WIDTH + nextX);

		int outUUL = nextZ + (nextY * LIGHT_BUFFER_WIDTH + currX);
		int outULL = nextZ + (currY * LIGHT_BUFFER_WIDTH + currX);
		int outUUR = nextZ + (nextY * LIGHT_BUFFER_WIDTH + nextX);
		int outULR = nextZ + (currY * LIGHT_BUFFER_WIDTH + nextX);

		float xFactor = fmod(translatedPos.x, 1.f);
		float yFactor = fmod(translatedPos.y, 1.f);
		float zFactor = fmod(translatedPos.z, 1.f);

		float combinedUpper = max(iPointers.lightImage[outDUL], iPointers.lightImage[outUUL]) * (1. - xFactor) + max(iPointers.lightImage[outDUR], iPointers.lightImage[outUUR]) * (xFactor);
		float combinedDown = max(iPointers.lightImage[outDLR], iPointers.lightImage[outULR]) * xFactor + max(iPointers.lightImage[outDLL], iPointers.lightImage[outULL]) * (1. - xFactor);
		float resultD = combinedUpper * yFactor + (1. - yFactor) * combinedDown;

		value = resultD;// *zFactor + resultD * (1. - zFactor);
		//value = lightImage[outDLL];
		return true;
	}
	return false;
}



__device__ bool worldPositionToTextureCoordinate(float3 position, int& out) {
	float3 translatedPos;
	bool res = getTranslatedPos(position, translatedPos);
	out = (((int)(translatedPos.z)) * LIGHT_BUFFER_WIDTH * LIGHT_BUFFER_WIDTH + ((int)translatedPos.y) * LIGHT_BUFFER_WIDTH + (int)(translatedPos.x));
	return res;
}



__device__ void traceMesh(const float3 currRayPos, const float3 currRayDir, const triangleMesh currMesh, float& closestDist, hitInfo& toReturn) {

	float tMin = 0;
	float tMax;

	float3 gridPos = (currRayPos - currMesh.bbMin) / currMesh.gridBoxDimensions;
	gridPos = make_float3(floor(gridPos.x), floor(gridPos.y), floor(gridPos.z));

	bool isAlreadyInside = max(gridPos.x, max(gridPos.y, gridPos.z)) < GRID_SIZE && min(gridPos.x, min(gridPos.y, gridPos.z)) >= 0;
	if (isAlreadyInside || (intersectBox(currRayPos, currRayDir, currMesh.bbMin, currMesh.bbMax, tMin, tMax) && tMin < closestDist && tMin > 0)) {

		// engage the GRID
		float3 currPos = currRayPos + (tMin + 0.001) * currRayDir;
		gridPos = (currPos - currMesh.bbMin) / currMesh.gridBoxDimensions;

		int stepsBeforeQuit = GRID_SIZE * 3;
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


					toReturn.normal = (1 - v - u) * currMesh.normals[currMesh.indices[iPos]] + u * currMesh.normals[currMesh.indices[iPos + 1]] + v * currMesh.normals[currMesh.indices[iPos + 2]];
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


__device__ hitInfo getHit(const float3 currRayPos, const float3 currRayDir) {
	float closestDist = 1000000;
	hitInfo toReturn;
	toReturn.hit = false;



	//float3 globalGridPos = (currRayPos - GLOBAL_GRID_MIN) / GLOBAL_GRID_DIMENSIONS;
	//globalGridPos = make_float3(floor(globalGridPos.x), floor(globalGridPos.y), floor(globalGridPos.z));
	//float tMin;
	//float tMax;

	//bool isAlreadyInsideGlobalGrid = max(globalGridPos.x, max(globalGridPos.y, globalGridPos.z)) < GLOBAL_GRID_SIZE && min(globalGridPos.x, min(globalGridPos.y, globalGridPos.z)) >= 0;
	//if (isAlreadyInsideGlobalGrid/* || (intersectBox(currRayPos, currRayDir, GLOBAL_GRID_MIN, GLOBAL_GRID_MAX, tMin, tMax) && tMin > 0)*/) {


		// mathematical objects
	for (int i = 0; i < iPointers.scene.numObjects; i++) {
		const objectInfo& curr = iPointers.scene.objects[i];
		float currDist;


		shapeInfo info = curr.shapeData;
		switch (curr.s) {
		case water: {
			shapeInfo otherInfo = info;
			otherInfo.normal = inverse(otherInfo.normal);
			float3 normalToUse = info.normal;
			bool needsToCommunicateInversion = false;
			bool intersected = intersectPlane(info, currRayPos, currRayDir, currDist);
			if (!intersected) {
				intersected = intersectPlane(otherInfo, currRayPos, currRayDir, currDist);
				normalToUse = otherInfo.normal;
				needsToCommunicateInversion = true;
			}

			if (intersected && currDist < closestDist) {
				closestDist = currDist;
				toReturn.info = curr.rayInfo;
				float3 pos = currRayPos + currDist * currRayDir;
				float3 waveInput = pos * 0.3 + make_float3(1 * currentTime + 10000, 10000, 10000);
				float strength = 0.5;
				float2 UVCoords = 0.05 * make_float2(pos.x, pos.z) + make_float2(0.1 * currentTime, 0) + make_float2(1000, 1000);//make_float2(fmod(abs(pos.x * 0.0001f), 1.f), fmod(abs(pos.z * 0.0001f), 1.f));//make_float2(fmod(abs(pos.x*0.01f + 1000.f), 0.99f), fmod(abs(pos.z*0.01f + 10000.f), 0.99f));

				float3 distortionSampled = sampleTexture(UVCoords, iPointers.waterNormal) * (1.0 / 255.0f) * 2 - make_float3(1., 1., 1.);

				float3 distortion = distortionSampled.x * make_float3(1, 0, 0) + distortionSampled.y * make_float3(0, 0, -1) + distortionSampled.z * make_float3(0, 1, 0);

				toReturn.normal = normalize(normalToUse + strength * distortion);
				toReturn.hit = true;
				toReturn.normalIsInversed = needsToCommunicateInversion;

			}

			toReturn.pos = currRayPos + closestDist * currRayDir;


			break;
		}
		case plane: {
			if (intersectPlane(info, currRayPos, currRayDir, currDist) && currDist < closestDist) {
				closestDist = currDist;
				toReturn.info = curr.rayInfo;
				toReturn.normal = info.normal;
				toReturn.hit = true;
				toReturn.pos = currRayPos + closestDist * currRayDir;

			}

			break;
		}
		case sphere: {
			if (length(info.pos - currRayPos) - info.rad < closestDist && intersectsSphere(currRayPos, currRayDir, info.pos, info.rad, currDist) && currDist < closestDist) {
				closestDist = currDist;
				float3 nextPos = currRayPos + currDist * currRayDir;
				toReturn.normal = normalize(nextPos - info.pos);
				toReturn.info = curr.rayInfo;
				toReturn.hit = true;
				toReturn.pos = currRayPos + closestDist * currRayDir;


				//float3 stuff =  sphericalCoordsToRectangular(toReturn.normal);
				//toReturn.info.color = stuff;


			}
			break;
		}
		}
	}


	// meshes
	for (int i = 0; i < iPointers.scene.numMeshes; i++) {
		traceMesh(currRayPos, currRayDir, iPointers.scene.meshes[i], closestDist, toReturn);
	}
	// BBM
	for (int i = 0; i < iPointers.scene.numBBMeshes; i++) {
		blackBoxMesh BBM = iPointers.scene.bbMeshes[i];

		float tMin = 0;
		float tMax;

		if (intersectBox(currRayPos, currRayDir, BBM.bbMin, BBM.bbMax, tMin, tMax) && tMin < closestDist && tMin > 0) {

			float3 nextPos = currRayPos + tMin  * currRayDir;

			//int index = rectangularCoordsToIndex(nextPos, currRayDir, BBM);
			//if (dot(currRayDir, ((BBM.bbMin + BBM.bbMax)*0.5) - nextPos) < 0)
			//	continue;
			BBMRes currBBMRes = rectangularCoordsToLerpedValue(nextPos, currRayDir, BBM);//BBM.texture[index];

			//currDist = length(currRayPos - currBBMRes.startP);


			if (currBBMRes.hitRatio >= 0.5) {
			//if (currBBMRes.hitRatio >= 0.01) {
				toReturn = hitInfo();
				closestDist = tMin;// length(currRayPos - currBBMRes.startP);
				toReturn.hit = true;
				//toReturn.info.reflectivity = currBBMRes.;
				//toReturn.info.refractivity = 0.0;//*/ 1. - currBBMRes.hitRatio;
				//toReturn.info.refractiveIndex = 1;
				//toReturn.info.color = currBBMRes.colorOut;// fromShort(currBBMRes.colorOut);// make_float3(1, 1, 1)* (10.f / length(currBBMRes.startP - BBM.center));//make_float3(1, 1, 1);
				//toReturn.pos = currBBMRes.startP;// fromShort(currBBMRes.startP);
				//toReturn.normal =  currBBMRes.startPNormal;
				toReturn.bbmHit = currBBMRes;
				toReturn.bbmHit.hitRatio = 1.0f;

			}
		}

	}


	return toReturn;
}

__device__ float getShadowTerm(const float3 originalPos, const float3 normal) {

#ifdef USING_PHOTON_MAPPED_SHADOWS

	float val;
	bool valid = worldPositionToLerpedValue(originalPos, val);
	if (valid) {
		return val * 0.01;
	}
	else {
		return 1.;
	}

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
	hitInfo hit = getHit(originalPos + 0.001 * toLightVec, toLightVec);
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
			toReturn = max(0., toReturn);
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
	prevHitInfo lastMaterialHit;
	prevHitInfo prevMaterialHit;
	float totalContributionRemaining = 0.0;
};

__device__ Ray make_ray(float3 pos, float3 dir, prevHitInfo lastHit, prevHitInfo prevHit, float remaining) {
	Ray toReturn;
	toReturn.currRayPos = pos;
	toReturn.currRayDir = dir;
	toReturn.lastMaterialHit = lastHit;
	toReturn.prevMaterialHit = prevHit;
	toReturn.totalContributionRemaining = remaining;
	return toReturn;

}

__device__  prevHitInfo getPrevMaterialHit(prevHitInfo Curr, prevHitInfo Last, prevHitInfo Prev) {
	return Curr.refractiveIndex != Last.refractiveIndex ? Last : Prev;
}

#define MAX_RAYS 5


enum class TracePassType {
	MAIN,
	LIGHT
};

__device__ float3 traceNonRecursive(const float3 initialRayPos, const float3 initialRayDir, int remainingDepth, const prevHitInfo prevHitToAddDepthFrom, const prevHitInfo prev2Hit, float totalContributionRemaining = 1.0, TracePassType type = TracePassType::MAIN) {

	Ray firstRay = make_ray(initialRayPos, initialRayDir, prevHitToAddDepthFrom, prev2Hit, totalContributionRemaining);
	float3 accumColor = make_float3(0, 0, 0);

	int currentNbrRays = 1;
	Ray AllRays[MAX_RAYS];
	AllRays[0] = firstRay;

	for (int i = 0; i < remainingDepth && currentNbrRays > 0; i++) {
		for (int j = 0; j < currentNbrRays; j++) {
			Ray currentRay = AllRays[j];

			hitInfo hit = getHit(currentRay.currRayPos, currentRay.currRayDir);

			float3 nextPos = hit.pos;
			float3 normal = hit.normal;
			float prevColorMP = 1 - powf(1. - currentRay.lastMaterialHit.insideColorDensity, length(nextPos - currentRay.currRayPos));

			if (!hit.hit) {
				accumColor = accumColor + currentRay.totalContributionRemaining * AIR_COLOR;
			}
			else if (hit.bbmHit.hitRatio >= 0.01) {
				accumColor = accumColor + prevColorMP * currentRay.lastMaterialHit.color * currentRay.totalContributionRemaining;
				currentRay.totalContributionRemaining *= (1. - prevColorMP);

				float3 color = currentRay.totalContributionRemaining * hit.bbmHit.colorOut * hit.bbmHit.hitRatio;
				float3 light_dir = STATIC_LIGHT_DIR;
				float angleFactor = (0. + 1.0 * max(0.0, dot(light_dir, hit.bbmHit.startPNormal)));
				accumColor = accumColor + ((0.8 * angleFactor + 0.2)  * color);

				if (hit.bbmHit.ray1Power > 0.01) {
					Ray nextRay = make_ray(hit.bbmHit.ray1Orig, hit.bbmHit.ray1Dir, currentRay.lastMaterialHit, currentRay.prevMaterialHit, currentRay.totalContributionRemaining *(1.f-hit.bbmHit.hitRatio));
					AllRays[currentNbrRays] = nextRay;
					currentNbrRays++;
				}
			}
			else {
				rayHitInfo info = hit.info;

				float3 reflected = make_float3(0, 0, 0);
				float3 refracted = make_float3(0, 0, 0);


				float extraReflection = 0;
				bool outside = dot(currentRay.currRayDir, normal) < 0;
				float3 refractBias = 0.002 * normal;
				refractBias = outside ? inverse(refractBias) : refractBias;
				float3 reflectBias = inverse(refractBias);


				float before = currentRay.totalContributionRemaining;
				accumColor = accumColor + prevColorMP * currentRay.lastMaterialHit.color * currentRay.totalContributionRemaining;
				currentRay.totalContributionRemaining *= (1. - prevColorMP);

				if (info.refractivity * currentRay.totalContributionRemaining > 0.001) {
					float kr = 1.0;

					// this gets the last material we passed through, for example if inside a glass submerged in water, it gets the water, need to remember the medium
					prevHitInfo PrevMaterialHit = getPrevMaterialHit(make_prevHitInfo(hit), currentRay.lastMaterialHit, currentRay.prevMaterialHit);
					prevHitInfo EnteringInfo = !outside || hit.normalIsInversed ? PrevMaterialHit : make_prevHitInfo(hit);

					fresnel(currentRay.currRayDir, normal, outside ? info.refractiveIndex / currentRay.prevMaterialHit.refractiveIndex : EnteringInfo.refractiveIndex / info.refractiveIndex, kr);

					if (kr < 1) {
						//if (currentNbrRays < MAX_RAYS) {
							float3 refractionDirection = normalize(refract(currentRay.currRayDir, normal, info.refractiveIndex));
							float3 refractionRayOrig = nextPos + refractBias;
							float refracMP = max(0., (1 - kr));
							Ray nextRay = make_ray(refractionRayOrig, refractionDirection, EnteringInfo, currentRay.lastMaterialHit, info.refractivity * refracMP * currentRay.totalContributionRemaining);
							AllRays[currentNbrRays] = nextRay;
							currentNbrRays++;
						//}

					}

					extraReflection = max(0.0, min(1., kr) * info.refractivity);
				}
				float reflecMP = (info.reflectivity + extraReflection) * currentRay.totalContributionRemaining;
				float3 reflectionOrig = nextPos + reflectBias;


				if (reflecMP > 0.001 && !(type == TracePassType::LIGHT)) {
					if (currentNbrRays < MAX_RAYS) {
						float3 reflectDir = reflect(currentRay.currRayDir, normal);
						Ray nextRay = make_ray(reflectionOrig, reflectDir, currentRay.lastMaterialHit, currentRay.prevMaterialHit, reflecMP);
						AllRays[currentNbrRays] = nextRay;
						currentNbrRays++;
					}
				}

				float colorMultiplier = max(0., (1. - max(0.f, info.reflectivity) - extraReflection - info.refractivity)) * currentRay.totalContributionRemaining;


				if (colorMultiplier > 0.001 && !(type == TracePassType::LIGHT)) {
					float3 color = colorMultiplier * info.color;
					float3 light_dir = STATIC_LIGHT_DIR;
					float angleFactor = (0. + 1.0 * max(0.0, dot(light_dir, normal)));
					float shadowFactor = getShadowTerm(reflectionOrig, normal);
					accumColor = accumColor + ((0.8 * shadowFactor * angleFactor + 0.2) * 1.0 * color);
				}
				else if (type == TracePassType::LIGHT) {

					float strength = max(0., (1. - max(0.f, info.reflectivity) - extraReflection - info.refractivity)) * 100 * before;
					float3 translatedPos;
					bool OK = getTranslatedPos(nextPos, translatedPos);
					if (OK) {

						int currY = floor(translatedPos.y);
						int currX = floor(translatedPos.x);
						int nextY = min(currY + 1, imageWidth - 1);
						int nextX = min(currX + 1, imageWidth - 1);
						int currZ = floor(translatedPos.z) * LIGHT_BUFFER_WIDTH * LIGHT_BUFFER_WIDTH;
						int nextZ = min(LIGHT_BUFFER_THICKNESS - 1., floor(translatedPos.z + 1)) * LIGHT_BUFFER_WIDTH * LIGHT_BUFFER_WIDTH;

						int outDUL = currZ + (nextY * LIGHT_BUFFER_WIDTH + currX);
						int outDLL = currZ + (currY * LIGHT_BUFFER_WIDTH + currX);
						int outDUR = currZ + (nextY * LIGHT_BUFFER_WIDTH + nextX);
						int outDLR = currZ + (currY * LIGHT_BUFFER_WIDTH + nextX);

						int outUUL = nextZ + (nextY * LIGHT_BUFFER_WIDTH + currX);
						int outULL = nextZ + (currY * LIGHT_BUFFER_WIDTH + currX);
						int outUUR = nextZ + (nextY * LIGHT_BUFFER_WIDTH + nextX);
						int outULR = nextZ + (currY * LIGHT_BUFFER_WIDTH + nextX);

						float xFactor = fmod(translatedPos.x, 1.f);
						float yFactor = fmod(translatedPos.y, 1.f);

						atomicAdd(&iPointers.lightImage[outDLL], strength * (1. - xFactor) * (1. - yFactor));
						atomicAdd(&iPointers.lightImage[outDUL], strength * (1. - xFactor) * (yFactor));
						atomicAdd(&iPointers.lightImage[outDUR], strength * (xFactor) * (yFactor));
						atomicAdd(&iPointers.lightImage[outDLR], strength * (xFactor) * (1. - yFactor));

						atomicAdd(&iPointers.lightImage[outULL], strength * (1. - xFactor) * (1. - yFactor));
						atomicAdd(&iPointers.lightImage[outUUL], strength * (1. - xFactor) * (yFactor));
						atomicAdd(&iPointers.lightImage[outUUR], strength * (xFactor) * (yFactor));
						atomicAdd(&iPointers.lightImage[outULR], strength * (xFactor) * (1. - yFactor));

					}

				}
			}
			AllRays[j] = AllRays[currentNbrRays - 1];
			currentNbrRays--;
		}
	}
	return accumColor;
}

__device__ void getXYZCoords(int& x, int& y, int& z) {

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int bt = blockDim.z;
	x = blockIdx.x * bw + tx;
	y = blockIdx.y * bh + ty;
	z = blockIdx.z + bt * tz;
}

__global__ void
cudaRender(inputPointers pointers, int imgw, int imgh, float currTime, inputStruct input)
{
	extern __shared__ uchar4 sdata[];

	int x, y, z;
	getXYZCoords(x, y, z);

	float3 forwardV = make_float3(input.forwardX, input.forwardY, input.forwardZ);
	float3 upV = make_float3(input.upX, input.upY, input.upZ);
	float3 rightV = normalize(cross(upV, forwardV));

	float sizeFarPlane = 10;
	float sizeNearPlane = sizeFarPlane * 0.5;
	float3 origin = make_float3(input.currPosX, input.currPosY, input.currPosZ);
	float distFarPlane = 4;
	float distFirstPlane = distFarPlane * 0.5;

	float3 center = make_float3(imgw / 2.0, imgh / 2.0, 0.);
	float3 distFromCenter = ((x - center.x) / imgw) * rightV + ((center.y - y) / imgw) * upV;
	startPos = (sizeNearPlane * distFromCenter) + origin + (distFirstPlane * forwardV);
	float3 secondPlanePos = (sizeFarPlane * distFromCenter) + (distFarPlane * forwardV) + origin;

	float3 dirVector = normalize(secondPlanePos - startPos);


	currentTime = currTime;
	pointers = pointers;
	imageWidth = imgw;
	imageHeight = imgh;

	prevHitInfo airMedium;
	airMedium.color = AIR_COLOR;
	airMedium.insideColorDensity = AIR_DENSITY;
	airMedium.refractiveIndex = 1.0;
	//float3 out = 255 * 3 * trace(firstPlanePos, dirVector, 10, input.beginMedium, 1.0);
	float3 out = 255 * 3 * traceNonRecursive(startPos, dirVector, 5, input.beginMedium, airMedium, 1.0);

	int firstPos = (y * imgw + x) * 4;
	pointers.image1[firstPos] = out.x;
	pointers.image1[firstPos + 1] = out.y;
	pointers.image1[firstPos + 2] = out.z;
}

__global__ void
cudaLightRender(inputPointers pointers, int imgw, int imgh, float currTime, inputStruct input)
{
	extern __shared__ uchar4 sdata[];

	int x, y, z;
	getXYZCoords(x, y, z);

	float3 forwardV = STATIC_LIGHT_DIR;
	float3 upV = make_float3(1, 0, 0);
	float3 rightV = normalize(cross(upV, forwardV));
	upV = cross(forwardV, rightV);

	float2 center = make_float2(imgw / 2.0, imgh / 2.0);
	float3 distFromCenter = ((x - center.x) / imgw) * rightV + ((center.y - y) / imgh) * upV;
	startPos = distFromCenter * LIGHT_PLANE_SIZE + forwardV * 400;
	float3 dirVector = inverse(forwardV);


	currentTime = currTime;
	iPointers = pointers;
	imageWidth = imgw;
	imageHeight = imgh;

	traceNonRecursive(startPos, dirVector, 6, input.beginMedium, input.beginMedium, 1.0, TracePassType::LIGHT);

}



__device__ BBMRes cudaBBMTrace(float3 initialRayPos, float3 initialRayDir, const triangleMesh bbmMesh) {
	BBMRes toReturn;

	Ray firstRay = make_ray(initialRayPos, initialRayDir, prevHitInfo(), prevHitInfo(), 1.f);
	float3 accumColor = make_float3(0, 0, 0);

	int currentNbrRays = 1;
	Ray AllRays[2];
	AllRays[0] = firstRay;
	for (int i = 0; i < 20 && currentNbrRays > 0; i++) {
		for (int j = 0; j < currentNbrRays; j++) {
			Ray currentRay = AllRays[j];

			hitInfo hit;
			float closestDist = 10000000;
			traceMesh(currentRay.currRayPos, currentRay.currRayDir, bbmMesh, closestDist, hit);
			currentNbrRays = 0;

			if (!hit.hit) {
				//toReturn.
				toReturn.hitRatio = 0.0f;

				goto beforeReturn;
				//break;
			}
			else {
				rayHitInfo info = hit.info;

				float3 reflected = make_float3(0, 0, 0);
				float3 refracted = make_float3(0, 0, 0);
				float3 normal = hit.normal;
				float3 nextPos = hit.pos;

				toReturn.hitRatio = 1.0f;
				toReturn.startP = nextPos;
				toReturn.colorOut = info.color;
				toReturn.startPNormal = normal;
				goto beforeReturn;

				//break;

				float extraReflection = 0;
				bool outside = dot(currentRay.currRayDir, normal) < 0;
				float3 refractBias = 0.002 * normal;
				refractBias = outside ? inverse(refractBias) : refractBias;
				float3 reflectBias = inverse(refractBias);


				float prevColorMP = 1 - powf(1. - currentRay.lastMaterialHit.insideColorDensity, length(nextPos - currentRay.currRayPos));
				accumColor = accumColor + prevColorMP * currentRay.lastMaterialHit.color * currentRay.totalContributionRemaining;
				currentRay.totalContributionRemaining *= (1. - prevColorMP);

				if (info.refractivity * currentRay.totalContributionRemaining > 0.001) {
					float kr = 1.0;

					// this gets the last material we passed through, for example if inside a glass submerged in water, it gets the water, need to remember the medium
					prevHitInfo PrevMaterialHit = getPrevMaterialHit(make_prevHitInfo(hit), currentRay.lastMaterialHit, currentRay.prevMaterialHit);
					prevHitInfo EnteringInfo = !outside || hit.normalIsInversed ? PrevMaterialHit : make_prevHitInfo(hit);

					fresnel(currentRay.currRayDir, normal, outside ? info.refractiveIndex / currentRay.prevMaterialHit.refractiveIndex : EnteringInfo.refractiveIndex / info.refractiveIndex, kr);

					if (kr < 1) {
						if (currentNbrRays < MAX_RAYS) {
							float3 refractionDirection = normalize(refract(currentRay.currRayDir, normal, info.refractiveIndex));
							float3 refractionRayOrig = nextPos + refractBias;
							float refracMP = max(0., (1 - kr));
							Ray nextRay = make_ray(refractionRayOrig, refractionDirection, EnteringInfo, currentRay.lastMaterialHit, info.refractivity * refracMP * currentRay.totalContributionRemaining);
							AllRays[currentNbrRays] = nextRay;
							currentNbrRays++;
						}

					}

					extraReflection = max(0.0, min(1., kr) * info.refractivity);
				}
				float reflecMP = (info.reflectivity + extraReflection) * currentRay.totalContributionRemaining;
				float3 reflectionOrig = nextPos + reflectBias;


				if ((info.reflectivity + extraReflection) > 0.33) {
					if (currentNbrRays < MAX_RAYS) {
						float3 reflectDir = reflect(currentRay.currRayDir, normal);
						Ray nextRay = make_ray(reflectionOrig, reflectDir, currentRay.lastMaterialHit, currentRay.prevMaterialHit, reflecMP);
						AllRays[currentNbrRays] = nextRay;
						currentNbrRays++;
					}
				}

				float colorMultiplier = max(0., (1. - max(0.f, info.reflectivity) - extraReflection - info.refractivity)) * currentRay.totalContributionRemaining;


				//if (colorMultiplier > 0.33) {
				float3 color = colorMultiplier * info.color;
				float3 light_dir = STATIC_LIGHT_DIR;
				float angleFactor = (0. + 1.0 * max(0.0, dot(light_dir, normal)));
				float shadowFactor = getShadowTerm(reflectionOrig, normal);
				accumColor = accumColor + ((0.8 * shadowFactor * angleFactor + 0.2) * 1.0 * color);
				//}
			}
			AllRays[j] = AllRays[currentNbrRays - 1];
			currentNbrRays--;
		}
	}
	beforeReturn:
	return toReturn;

}


__global__ void
cudaBBMRender(BBMPassInput input) {
	int x, y, z;
	getXYZCoords(x, y, z);

	BBMRes toReturn;


	int directionIndex = z / (input.bbm.angleResolution * input.bbm.angleResolution);
	float3 direction = /*make_float3(0, 0, -1);//*/intToDirection(directionIndex);

	float3 center = (input.bbm.bbMin + input.bbm.bbMax)*0.5f;
	float3 diachongus = input.bbm.bbMax - input.bbm.bbMin;

	float3 tan =/* make_float3(0, 1, 0);//*/ getTan(direction);
	float3 biTan = /*make_float3(1, 0, 0);//*/ cross(direction, tan);
	tan = dot(tan, diachongus) > 0 ? tan : inverse(tan);
	biTan = dot(biTan, diachongus) > 0 ? biTan : inverse(biTan);
	float3 dirDistFromBBMin = (direction * diachongus) * (dot(direction, diachongus) > 0 ? 0.001 : 1.001);

	float3 stepTan = tan * diachongus / ((float)input.bbm.sideResolution - 1);
	float3 stepBiTan = biTan * diachongus /((float)input.bbm.sideResolution - 1);

	float3 offsetTan = stepTan * x;
	float3 offsetBiTan = stepBiTan * y;
	float3 initialRayPos = (input.bbm.bbMin - dirDistFromBBMin) +offsetTan + offsetBiTan;



	int adjustedY = (z % (input.bbm.angleResolution * input.bbm.angleResolution)) / input.bbm.angleResolution;
	int adjustedZ = (z % input.bbm.angleResolution);
	float stepLen = PI / MAX(1,input.bbm.angleResolution - 1);

	float yRot = -PI * 0.5f + stepLen * adjustedY;
	float zRot = -PI * 0.5f + stepLen * adjustedZ;

	if (input.bbm.angleResolution == 1) {
		yRot = 0;
		zRot = 0;
	}

	//
	float3 lookingAtCenter = normalize(center - initialRayPos);
	float3 lookingTan = getTan(lookingAtCenter);
	float3 lookingBiTan = cross(lookingAtCenter, lookingTan);
	lookingTan = dot(lookingTan, diachongus) > 0 ? lookingTan : inverse(lookingTan);
	lookingBiTan = dot(lookingBiTan, diachongus) > 0 ? lookingBiTan : inverse(lookingBiTan);

	float3 stepLookingTan = lookingTan;// *diachongus / ((float)input.bbm.sideResolution - 1);
	float3 stepLookingBiTan = lookingBiTan;// *diachongus / ((float)input.bbm.sideResolution - 1);
	//

	float3 lookDir = rotateAngleAxis(direction, yRot, tan);
	lookDir = rotateAngleAxis(lookDir, zRot, biTan);
	

	// result is how long we should travel in each tan and bitan directions



	BBMRes total;
	int numberOfHits = 0;
	// super sample for both angle and point, we only have to do this for setup so doesn't cost any realtime performance
	int kernelDiameter = 5;
	float totalSamples = powf(kernelDiameter, 4);
	int maxIndex = (kernelDiameter  - 1)/2;
	float offset = 0.5f / MAX(maxIndex, 1);
	for (float i = -maxIndex; i <= maxIndex; i++) {
		for (float j = -maxIndex; j <= maxIndex; j++) {
			for (float k = -maxIndex; k <= maxIndex; k++) {
				for (float l = -maxIndex; l <= maxIndex; l++) {

				float3 lookDirCurr = rotateAngleAxis(direction, ((float)(i* offset)) * stepLen + yRot, tan);
				lookDirCurr = rotateAngleAxis(lookDirCurr, ((float)(j* offset)) * stepLen + zRot, biTan);
				float3 initialPos = initialRayPos + (k * offset * stepTan) + (l* offset * stepBiTan);

				BBMRes curr = cudaBBMTrace(initialPos, lookDirCurr, input.mesh);
				if (curr.hitRatio > 0.1f) {
					total = total + curr;
					++numberOfHits;
				}
				}
			}
		}
	}
	float hitRatio = numberOfHits > 0 ? (1.0f / numberOfHits) : 0.0f;
	toReturn = total* hitRatio;
	toReturn.hitRatio = /*1.0f; //*/numberOfHits / totalSamples;
	// store our result for later use
	int index = rectangularCoordsToIndex(initialRayPos, lookDir, input.bbm);
	//int index = getIndex(directionIndex, x, y, adjustedY, adjustedZ, input.bbm);

	input.bbm.texture[index] = toReturn;
}

__global__ void
cudaClear(unsigned int* buffer, int imgw)
{
	extern __shared__ uchar4 sdata[];

	int x, y, z;
	getXYZCoords(x, y, z);

	int firstPos = (z * (imgw * imgw) + y * imgw + x);
	buffer[firstPos] = 0;
}

extern "C" void
launch_cudaLight(dim3 grid, dim3 block, int sbytes, inputPointers pointers, int imgw, int imgh, float currTime, inputStruct input)
{

	cudaLightRender << < grid, block, sbytes >> > (pointers, imgw, imgh, currTime, input);
}

extern "C" void
launch_cudaClear(dim3 grid, dim3 block, int sbytes, int imgw, unsigned int* buffer)
{

	cudaClear << < grid, block, sbytes >> > (buffer, imgw);
}



extern "C" void
launch_cudaRender(dim3 grid, dim3 block, int sbytes, inputPointers pointers, int imgw, int imgh, float currTime, inputStruct input)
{

	cudaRender << < grid, block, sbytes >> > (pointers, imgw, imgh, currTime, input);
}


extern "C" void
launch_cudaBBMRender(dim3 grid, dim3 block, int sbytes, BBMPassInput input)
{

	cudaBBMRender << < grid, block, sbytes >> > (input);
}



