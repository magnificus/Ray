#pragma once
#include "rayHelpers.cu"



cudaError_t cuda();
__global__ void kernel(){
  
}

__device__ bool intersectsSphere(const float3 &origin, const float3& dir,  const shapeInfo& info, float &t) {

		float t0, t1; // solutions for t if the ray intersects 

		float rad2 = powf(info.rad, 2);

		float3 L = info.pos - origin;
		float tca = dot(dir, L);
		 //if (tca < 0) return false;
		float d2 = dot(L,L) - tca * tca;
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
	if (denom > 1e-6) {
		float3 p0l0 = p.pos - l0;
		t = dot(p0l0, p.normal) / denom;
		return (t >= 0);
	}
	return false;
}


__device__ bool rayTriangleIntersect(
	const float3& orig, const float3& dir,
	const float3& v0, const float3& v1, const float3& v2,
	float& t, float& u, float& v)
{
	float3 v0v1 = v1 - v0;
	float3 v0v2 = v2 - v0;
	float3 pvec = cross(dir, v0v2);
	float det = dot(pvec,v0v1);
	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle
	float kEpsilon = 0.000001;
	if (det < kEpsilon) return false;
	float invDet = 1 / det;

	float3 tvec = orig - v0;
	u = dot(tvec,pvec) * invDet;
	if (u < 0 || u > 1) return false;

	float3 qvec = cross(tvec,v0v1);
	v = dot(dir,qvec) * invDet;
	if (v < 0 || u + v > 1) return false;

	t = dot(v0v2,qvec) * invDet;

	return true;
}


__device__ void fresnel(const float3& I, const float3& N, const float& ior, float& kr)
{
	float cosi = clamp(-1, 1, dot(I, N));
	float etai = 1, etat = ior;
	if (cosi > 0) { float temp = etai; etai = etat; etat = temp;}
	// Compute sini using Snell's law
	float sint = etai / etat * sqrtf(max(0.f, 1 - cosi * cosi));
	// Total internal reflection
	if (sint >= 1) {
		kr = 1;
	}
	else {
		float cost = sqrtf(max(0.f, 1 - sint * sint));
		cosi = abs(cosi);
		float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
		float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
		kr = (Rs * Rs + Rp * Rp) / 2;
	}

}


__device__ float3 refract(const float3& I, const float3& N, const float& ior)
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

__device__ float3 reflect(const float3& I, const float3& N)
{
	return I - 2 * dot(I, N) * N;
}

struct hitInfo {
	int objectIndex = -1;
	float3 pos;
	float3 normal;

};

#define LIGHT_POS make_float3(0,5,20)


__device__ hitInfo getHit(float3 currRayPos, float3 currRayDir, const float& currTime, const objectInfo* objects, int numObjects) {
	float closestDist = 1000000;
	float3 normal;
	hitInfo toReturn;
	int closestObjectIndex = -1;


	for (int i = 0; i < numObjects; i++) {
		const objectInfo& curr = objects[i];
		float currDist;

		switch (curr.s) {
		case plane: {
			shapeInfo p1 = curr.shapeData;
			if (intersectPlane(p1, currRayPos, currRayDir, currDist) && currDist < closestDist) {
				closestDist = currDist;
				closestObjectIndex = i;

				normal = p1.normal;
			}

			break;
		}
		case sphere: {
			shapeInfo s1 = curr.shapeData;
			if (intersectsSphere(currRayPos, currRayDir, s1, currDist) && currDist < closestDist) {
				closestDist = currDist;
				closestObjectIndex = i;

				float3 nextPos = currRayPos + currDist * currRayDir;
				normal = normalize(nextPos - s1.pos);

			}
			break;
		}
		}
	}

	toReturn.objectIndex = closestObjectIndex;
	toReturn.normal = normal;
	toReturn.pos = currRayPos + closestDist * currRayDir;
	return toReturn;
}


__device__ float getShadowTerm(const float3 originalPos, const float currTime, const objectInfo* objects, int numObjects) {
	float3 toLightVec = normalize(LIGHT_POS - originalPos);
	hitInfo hit = getHit(originalPos, toLightVec, currTime, objects, numObjects);

	if (hit.objectIndex == -1 || length(hit.pos - originalPos) > length(originalPos - LIGHT_POS)) {
		return 1.;
	}
	return objects[hit.objectIndex].refractivity * 0.8 + 0.2;

}

__device__ float3 trace(const float3 currRayPos, const float3 currRayDir, int remainingDepth, const float currTime, objectInfo *objects, int numObjects) {
	if (remainingDepth <= 0) {
		return make_float3(0,0,0);
	}
	
	float3 v1 = make_float3(0, 0, 10);
	float3 v2 = make_float3(10, 0, 10);
	float3 v3 = make_float3(10, 10, 10);

	float t;
	float u;
	float v;

	bool hitTri = rayTriangleIntersect(currRayPos, currRayDir, v3, v2, v1, t, u, v);

	float3 hitPosTri = currRayPos + t * currRayDir;

	//if (hitTri) {
	//	return make_float3(1, 0, 0);
	//}
	//else {
	//	return make_float3(0, 0, 0);
	//}



	hitInfo hit = getHit(currRayPos, currRayDir, currTime, objects, numObjects);

	if (hit.objectIndex == -1) {
		return make_float3(0,0,0);
	}
	else {

		objectInfo currObject = objects[hit.objectIndex];
		float3 reflected = make_float3(0, 0, 0);
		float3 refracted = make_float3(0, 0, 0);
		float3 nextPos = hit.pos;
		float3 normal = hit.normal;

		if (hitTri && length(hitPosTri - currRayPos) < length(nextPos - currRayPos)) {
			return make_float3(1, 0, 0);
		}

		float extraReflection = 0;
		float3 bias = 0.001 * normal;
		if (currObject.refractivity > 0.) {
			float kr;
			bool outside = dot(currRayDir, normal) < 0;
			fresnel(currRayDir, normal, outside? currObject.refractiveIndex : 1 / currObject.refractiveIndex, kr);


			if (kr <= 1) {
				float3 refractionDirection = normalize(refract(currRayDir, normal, currObject.refractiveIndex));
				float3 refractionRayOrig = outside ? nextPos - bias : nextPos + bias;
				refracted = currObject.refractivity *(1-kr)* trace(refractionRayOrig, refractionDirection, remainingDepth - 1, currTime, objects, numObjects);
			}
			extraReflection = min(1.,kr) * currObject.refractivity;

		}
		if (currObject.reflectivity + extraReflection > 0.) {
			float3 reflectDir = reflect(currRayDir, normal);
			reflected = (currObject.reflectivity + extraReflection )* trace(nextPos + bias, reflectDir, remainingDepth - 1, currTime, objects, numObjects);
		}
		float3 color = (1 - currObject.reflectivity - extraReflection - currObject.refractivity) * currObject.color;
		return 1000 * (1 / powf(length(nextPos - LIGHT_POS), 2)) * getShadowTerm(nextPos + bias, currTime, objects, numObjects) * color + reflected + refracted;
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
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;

	float3 forwardV = make_float3(input.forwardX, input.forwardY, input.forwardZ);
	float3 upV = make_float3(input.upX, input.upY, input.upZ);
	float3 rightV = normalize(cross(upV,forwardV));

	float sizeFarPlane = 10;
	float sizeNearPlane = sizeFarPlane *0.5;
	float3 origin = make_float3(input.currPosX, input.currPosY, input.currPosZ);
	float distFarPlane = 4;
	float distFirstPlane = distFarPlane *0.5;

	float3 center = make_float3(imgw / 2.0, imgh / 2.0, 0.);
	float3 distFromCenter = ((x - center.x) / imgw) * rightV + ((center.y - y) / imgh) * upV;
	float3 firstPlanePos = (sizeNearPlane*distFromCenter) + origin + (distFirstPlane * forwardV);
	float3 secondPlanePos = (sizeFarPlane * distFromCenter) + (distFarPlane * forwardV) + origin;

	float3 dirVector = normalize(secondPlanePos - firstPlanePos);
	float3 out = 255*trace(firstPlanePos, dirVector, 5, currTime, pointers.objects, pointers.numObjects);


	pointers.g_odata[y * imgw + x] = rgbToInt(out.x, out.y, out.z);
}
extern "C" void
launch_cudaRender(dim3 grid, dim3 block, int sbytes, inputPointers pointers, int imgw, int imgh, float currTime,inputStruct input)
{

	cudaRender << < grid, block, sbytes >> >(pointers, imgw, imgh, currTime, input);
}

