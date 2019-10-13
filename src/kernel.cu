#pragma once
#include "rayHelpers.cu"
#include "perlin.h"
#include "cuda.h"



#define USING_SHADOWS
#define USING_DOUBLE_TAP_SHADOWS
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
__device__ sceneInfo *scene;
__device__ unsigned int* lightImage;
__device__ int imageWidth;
__device__ int imageHeight;
//sceneInfo info;


__device__ bool intersectsSphere(const float3& origin, const float3& dir, const float3 pos, const float rad, float& t) {

	float t0, t1; // solutions for t if the ray intersects 

	float rad2 = powf(rad, 2);

	float3 L = pos - origin;
	float tca = dot(dir, L);
	//if (tca < 0) return false;
	float d2 = dot(L, L) - tca * tca;
	if (d2 > rad2) return false;
	float thc = sqrt(rad2 - d2);
	t0 = tca - thc;
	t1 = tca + thc;

	if (t0 > t1) {
		float temp = t0;
		t0 = t1;
		t1 = temp;
	}

	if (t0 < 0) {
		t0 = t1; // if t0 is negative, let's use t1 instead 
		if (t0 < 0) return false; // both t0 and t1 are negative 
	}
	t = t0;
	return true;
}

// plane normal, plane point, ray start, ray dir, point along line
__device__ bool intersectPlane(const shapeInfo& p, const float3& l0, const float3& l, float& t)
{
	// assuming vectors are all normalized
	float denom = dot(p.normal, l);
	if (denom < -1e-8) {
		float3 p0l0 = p.pos - l0;
		t = dot(p0l0, p.normal) / denom;
		return (t >= 0);
	}
	return false;
}

__device__ bool rayTriangleIntersect(
	float3 orig, float3 dir, float3 v0, const float3& v1, const float3& v2,
	float& t, float& u, float& v)
{
	// compute plane's normal
	float3 v0v1 = v1 - v0;
	float3 v0v2 = v2 - v0;

	//// no need to normalize
	float3 N = cross(v0v1, v0v2); // N 
	float denom = dot(N, N);


	//// Step 1: finding P

	// check if ray and plane are parallel ?
	float NdotRayDirection = dot(N, dir);
	if (fabs(NdotRayDirection) < 0.0001) // almost 0 
		return false; // they are parallel so they don't intersect ! 

	// compute d parameter using equation 2
	float d = dot(N, v0);

	// compute t (equation 3)
	t = (dot(N, orig) + d) / NdotRayDirection;
	// check if the triangle is in behind the ray
	if (t < 0) return false; // the triangle is behind 

	// compute the intersection point using equation 1
	float3 P = orig + t * dir;

	// Step 2: inside-outside test
	float3 C; // vector perpendicular to triangle's plane 

	// edge 0
	float3 edge0 = v1 - v0;
	float3 vp0 = P - v0;
	C = cross(edge0, vp0);
	if (dot(N, C) < 0) return false; // P is on the right side 

	// edge 1
	float3 edge1 = v2 - v1;
	float3 vp1 = P - v1;
	C = cross(edge1, vp1);
	if ((u = dot(N, C)) < 0)  return false; // P is on the right side 

	// edge 2
	float3 edge2 = v0 - v2;
	float3 vp2 = P - v2;
	C = cross(edge2, vp2);
	if ((v = dot(N, C)) < 0) return false; // P is on the right side; 

	u /= denom;
	v /= denom;

	return true; // this ray hits the triangle 
}




__device__ bool RayIntersectsTriangle(float3 rayOrigin,
	float3 rayVector,
	float3 vertex0, float3 vertex1, float3 vertex2,
	float& t, float& u, float& v)
{

	const float EPSILON = 0.001;
	float3 edge1, edge2, h, s, q;
	float a, f;
	edge1 = vertex1 - vertex0;
	edge2 = vertex2 - vertex0;
	h = cross(rayVector, edge2);
	a = dot(edge1, h);
	if (a > -EPSILON && a < EPSILON)
		return false;    // This ray is parallel to this triangle.
	f = 1.0 / a;
	s = rayOrigin - vertex0;
	u = f * dot(s, h);
	if (u < 0.0 || u > 1.0)
		return false;
	q = cross(s, edge1);
	v = f * dot(rayVector, q);
	if (v < 0.0 || u + v > 1.0)
		return false;
	// At this stage we can compute t to find out where the intersection point is on the line.
	t = f * dot(edge2, q);

	return t > EPSILON && !((u < 0.0 || u > 1.0) || (v < 0.0 || u + v > 1.0));
}


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


	return (otherDir1 * d1 + otherDir2 * d2);

}

__device__ bool getTranslatedPos(float3 position, float3 &translatedPos) {
	float3 beforeTranslation = make_float3(LIGHT_BUFFER_WORLD_RATIO * position.x, LIGHT_BUFFER_WORLD_RATIO * position.z, LIGHT_BUFFER_THICKNESS_WORLD_RATIO * (position.y +50));
	translatedPos = beforeTranslation + make_float3(0.5, 0.5, 0.5);
	translatedPos = translatedPos * make_float3(LIGHT_BUFFER_WIDTH, LIGHT_BUFFER_WIDTH, LIGHT_BUFFER_THICKNESS);
	if (translatedPos.x >= 0 && translatedPos.x < LIGHT_BUFFER_WIDTH && translatedPos.y >= 0 && translatedPos.y < LIGHT_BUFFER_WIDTH && translatedPos.z >= 0 && translatedPos.z < LIGHT_BUFFER_THICKNESS) {
	//if(max(max(fabs(beforeTranslation.x), fabs(beforeTranslation.y)), fabs(beforeTranslation.z)) < LIGHT_BUFFER_WORLD_SIZE / 2){
		return true;
	}
	return false;
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
		float yFactor = translatedPos.y - floor(translatedPos.y);
		float combinedUpper = lightImage[outUL] * (1.-xFactor) + lightImage[outUR] * (xFactor);
		float combinedDown = lightImage[outLR] * xFactor + lightImage[outLL] * (1. - xFactor);
		float result = combinedUpper* yFactor + (1. - yFactor) * combinedDown;
		value = result;
		return true;
	}
	return false;
}



//__device__ bool worldPositionToTextureCoordinate(float3 position, int& out) {
//	float3 translatedPos;
//	bool res = getTranslatedPos(position, translatedPos);
//	out = (((int)(translatedPos.z)) * LIGHT_BUFFER_WIDTH*LIGHT_BUFFER_WIDTH + ((int)translatedPos.y) * LIGHT_BUFFER_WIDTH + (int)(translatedPos.x));
//	return res;
//}


__device__ hitInfo getHit(const float3 currRayPos,const float3 currRayDir) {
	float closestDist = 1000000;
	float3 normal;
	hitInfo toReturn;
	toReturn.hit = false;


	// mathematical objects
	for (int i = 0; i < scene->numObjects; i++) {
		const objectInfo& curr = scene->objects[i];
		float currDist;


		shapeInfo info = curr.shapeData;
		if (info.isMoving) {
			info.pos = make_float3(sin(currentTime*0.3) * info.pos.x, info.pos.y, cos(currentTime*0.3) * info.pos.z);

		}
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
				float3 waveInput = (currRayPos + currDist * currRayDir) * 0.3 + make_float3(1 * currentTime + 10000, 10000, 10000);
				float strength = 2000;

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
			if (intersectsSphere(currRayPos, currRayDir, info.pos, info.rad, currDist) && currDist < closestDist) {
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
	for (int i = 0; i < scene->numMeshes; i++) {
		triangleMesh currMesh = scene->meshes[i];

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
			while (--stepsBeforeQuit >= 0 && max(gridPos.x, max(gridPos.y, gridPos.z)) < GRID_SIZE && min(gridPos.x, min(gridPos.y, gridPos.z)) >= 0) {

				gridPos = make_float3(floor(gridPos.x), floor(gridPos.y), floor(gridPos.z));
				unsigned int gridPosLoc = GRID_POS(gridPos.x, gridPos.y, gridPos.z);

				float t;
				float u;
				float v;
				for (unsigned int j = 0; j < currMesh.gridSizes[gridPosLoc]; j++) {
					unsigned int iPos = currMesh.grid[gridPosLoc][j];
					bool hitTriangle = RayIntersectsTriangle(currRayPos, currRayDir, currMesh.vertices[currMesh.indices[iPos]], currMesh.vertices[currMesh.indices[iPos + 1]], currMesh.vertices[currMesh.indices[iPos + 2]], t, u, v);
					if (hitTriangle && t < closestDist) {
						closestDist = t;
						toReturn.info = currMesh.rayInfo;

						normal = (1 - v - u)* currMesh.normals[currMesh.indices[iPos]] + u * currMesh.normals[currMesh.indices[iPos + 1]] + v * currMesh.normals[currMesh.indices[iPos + 2]];
						toReturn.hit = true;
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
	hitInfo hit = getHit(originalPos + 0.01 *toLightVec, toLightVec);
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
			hit = getHit(hit.pos + 0.01 * toLightVec, toLightVec);
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

//#if __CUDA_ARCH__ < 600
//__device__ double atomicAdd(double* address, double val)
//{
//	unsigned long long int* address_as_ull =
//		(unsigned long long int*)address;
//	unsigned long long int old = *address_as_ull, assumed;
//
//	do {
//		assumed = old;
//		old = atomicCAS(address_as_ull, assumed,
//			__double_as_longlong(val +
//				__longlong_as_double(assumed)));
//
//		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//	} while (assumed != old);
//
//	return __longlong_as_double(old);
//}
//#endif



__device__ float3 trace(const float3 currRayPos, const float3 currRayDir, int remainingDepth, const hitInfo &prevHitToAddDepthFrom, float totalContributionRemaining = 1.0, bool isLightPass = false) {

	hitInfo hit = getHit(currRayPos, currRayDir);

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
		float3 reflectBias = 0.001 * normal;
		float prevColorMP = 0;
		float3 extraPrevColor = make_float3(0,0,0);
		bool outside = dot(currRayDir, normal) < 0;

		if (prevHitToAddDepthFrom.info.insideColorDensity > 0.001) {
			prevColorMP = 1 - powf(1. - prevHitToAddDepthFrom.info.insideColorDensity, length(nextPos - currRayPos)+1);
			extraPrevColor = prevColorMP * prevHitToAddDepthFrom.info.color;
		}

		if (prevColorMP > 0.999 || remainingDepth == 1 || totalContributionRemaining < 0.01)
			return info.color * (1. - prevColorMP) + extraPrevColor;

		if (info.refractivity* totalContributionRemaining > 0.001) {
			float kr = 1.0;;
			fresnel(currRayDir, normal, outside ? info.refractiveIndex : 1 / info.refractiveIndex, kr);


			if (kr < 1) {
				float3 refractionDirection = normalize(refract(currRayDir, normal, info.refractiveIndex));
				float3 refractionRayOrig = outside ? nextPos - refractBias : nextPos + refractBias;

				float refracMP = max(0., (1 - kr));
				refracted = info.refractivity * refracMP * trace(refractionRayOrig, refractionDirection, remainingDepth - 1,  outside ^ hit.normalIsInversed ? hit : hitInfo(), totalContributionRemaining* refracMP, isLightPass);
			}
			extraReflection = max(0.,min(1., kr) * info.refractivity);

		}
		if ((info.reflectivity + extraReflection)* totalContributionRemaining > 0.001 && !isLightPass) {
			float3 reflectDir = reflect(currRayDir, normal);
			float3 reflectionOrig = outside ? nextPos + reflectBias : nextPos - reflectBias;
			float reflecMP = info.reflectivity + extraReflection;

			reflected = reflecMP * trace(reflectionOrig, reflectDir, remainingDepth - 1, prevHitToAddDepthFrom, reflecMP*totalContributionRemaining, isLightPass);
		}

		float colorMultiplier = max(0., (1. - info.reflectivity - extraReflection - info.refractivity));
		if (!isLightPass) {
			float3 color = colorMultiplier * info.color;
			float3 light_dir = STATIC_LIGHT_DIR;
			float angleFactor = (0.0 + 1.0 * max(0.0, dot(light_dir, normal)));
			float shadowFactor = 0;
			//if (colorMultiplier * (1.-prevColorMP) > 0.1) {
			shadowFactor = getShadowTerm(nextPos + 0.01 * inverse(currRayDir), normal);
			//}
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
	scene = &pointers.scene;
	lightImage = pointers.lightImage;
	imageWidth = imgw;
	imageHeight = imgh;
	float3 out = 255 * 3 * trace(firstPlanePos, dirVector, 10, input.beginMedium, 1.0);


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

	float3 forwardV = STATIC_LIGHT_DIR;
	float3 upV = make_float3(1,0,0);
	float3 rightV = normalize(cross(upV, forwardV));
	upV = cross(forwardV, rightV);

	float2 center = make_float2(imgw / 2.0, imgh / 2.0);
	float3 distFromCenter = ((x - center.x) / imgw) * rightV + ((center.y - y) / imgh) * upV;
	float3 startPos = distFromCenter * LIGHT_PLANE_SIZE + STATIC_LIGHT_DIR * 400 ;
	float3 dirVector = inverse(STATIC_LIGHT_DIR);

	//float3 distFromCenter = make_float3(((x - center.x) / imgw), 0, ((center.y - y) / imgh));
	//float3 startPos = distFromCenter * LIGHT_PLANE_SIZE + make_float3(0, 100, 0);
	//float3 dirVector = make_float3(0,-1,0);

	currentTime = currTime;
	scene = &pointers.scene;
	lightImage = pointers.lightImage;
	imageWidth = imgw;
	imageHeight = imgh;
	trace(startPos, dirVector, 10, hitInfo(), 1.0, true);

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


