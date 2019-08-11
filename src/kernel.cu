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
	const float3& orig, const float3& dir, const float3& N,
	const float3& v0, const float3& v1, const float3& v2,
	float& t, float& u, float& v)
{
	// compute plane's normal
	float3 v0v1 = v1 - v0;
	float3 v0v2 = v2 - v0;
	// no need to normalize
	//float3 N = cross(v0v1, v0v2); // N 
	float denom = dot(N, N);


	if (!intersectPlane(make_shapeInfo(v1, normalize(N), 0), orig, dir, t)) {
		return false;
	}

	//// Step 1: finding P

	//// check if ray and plane are parallel ?
	//float NdotRayDirection = dot(N,dir);
	////if (fabs(NdotRayDirection) < 0.000001) // almost 0 
	////	return false; // they are parallel so they don't intersect ! 

	//// compute d parameter using equation 2
	////N = normalize(N);
	//float d = dot(N,v0);

	//// compute t (equation 3)
	//t = (dot(N, orig) + d) / NdotRayDirection;
	//// check if the triangle is in behind the ray
	//if (t < 0) return false; // the triangle is behind 


	//return true;
	// compute the intersection point using equation 1
	float3 P = orig + t * dir;

	// Step 2: inside-outside test
	float3 C; // vector perpendicular to triangle's plane 

	// edge 0
	float3 edge0 = v1 - v0;
	float3 vp0 = P - v0;
	C = cross(edge0,vp0);
	if (dot(N,C) < 0) return false; // P is on the right side 

	// edge 1
	float3 edge1 = v2 - v1;
	float3 vp1 = P - v1;
	C = cross(edge1,vp1);
	if ((u = dot(N,C)) < 0)  return false; // P is on the right side 

	// edge 2
	float3 edge2 = v0 - v2;
	float3 vp2 = P - v2;
	C = cross(edge2,vp2);
	if ((v = dot(N,C)) < 0) return false; // P is on the right side; 

	u /= denom;
	v /= denom;

	return true; // this ray hits the triangle 
}



//__device__ bool rayTriangleIntersect(
//	const float3& orig, const float3& dir,
//	const float3& v0, const float3& v1, const float3& v2,
//	float& t, float& u, float& v)
//{
//	float3 v0v1 = v1 - v0;
//	float3 v0v2 = v2 - v0;
//	float3 pvec = cross(dir, v0v2);
//	float det = dot(pvec,v0v1);
//	// if the determinant is negative the triangle is backfacing
//	// if the determinant is close to 0, the ray misses the triangle
//	float kEpsilon = 0.000001;
//	if (det < kEpsilon) return false;
//	float invDet = 1 / det;
//
//	float3 tvec = orig - v0;
//	u = dot(tvec,pvec) * invDet;
//	if (u < 0 || u > 1) return false;
//
//	float3 qvec = cross(tvec,v0v1);
//	v = dot(dir,qvec) * invDet;
//	if (v < 0 || u + v > 1) return false;
//
//	t = dot(v0v2,qvec) * invDet;
//
//	return true;
//}
//

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
	bool hitMesh = false;
	float3 pos;
	float3 normal;

};

#define LIGHT_POS make_float3(0,5,20)


__device__ hitInfo getHit(float3 currRayPos, float3 currRayDir, const sceneInfo& scene) {
	float closestDist = 1000000;
	float3 normal;
	hitInfo toReturn;
	int closestObjectIndex = -1;


	// mathematical objects
	for (int i = 0; i < scene.numObjects; i++) {
		const objectInfo& curr = scene.objects[i];
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


	// meshes
	for (int i = 0; i < scene.numMeshes; i++) {
		const triangleMesh currMesh = scene.meshes[i];
		for (unsigned int j = 0; j < scene.meshes[i].numIndices; j+=3) {
			float t;
			float u;
			float v;
			bool hitTriangle = rayTriangleIntersect(currRayPos, currRayDir, currMesh.normals[currMesh.indices[j]], currMesh.vertices[currMesh.indices[j]], currMesh.vertices[currMesh.indices[j + 1]], currMesh.vertices[currMesh.indices[j + 2]], t, u, v);
			if (hitTriangle && t < closestDist) {
				closestDist = t;
				toReturn.hitMesh = true;
				normal = u*currMesh.normals[currMesh.indices[j]] + v* currMesh.normals[currMesh.indices[j+1]] +(1-v-u)* currMesh.normals[currMesh.indices[j+2]];
			//	closestObjectIndex = 0;

			//	//c
			}


		}

	}


	toReturn.objectIndex = closestObjectIndex;
	toReturn.normal = normal;
	toReturn.pos = currRayPos + closestDist * currRayDir;
	return toReturn;
}


__device__ float getShadowTerm(const float3 originalPos, const sceneInfo& scene) {
	float3 toLightVec = normalize(LIGHT_POS - originalPos);
	hitInfo hit = getHit(originalPos, toLightVec, scene);

	if (hit.objectIndex == -1 || length(hit.pos - originalPos) > length(originalPos - LIGHT_POS)) {
		return 1.;
	}
	return 0.2;
	//return objects[hit.objectIndex].refractivity * 0.8 + 0.2;

}


__device__ float3 trace(const float3 currRayPos, const float3 currRayDir, int remainingDepth, const sceneInfo &scene) {
	if (remainingDepth <= 0) {
		return make_float3(0,0,0);
	}


	//int blargh;
	//for (int i = 0; i < 1000; i++) {
	//	float t;
	//	blargh += intersectsSphere(currRayPos, currRayDir, scene.objects[0].shapeData, t);
	//}
	//return make_float3(0, 0, blargh);


	hitInfo hit = getHit(currRayPos, currRayDir, scene);


	//if (hit.hitMesh)
	//	return make_float3(0, 1, 0);

	if (hit.objectIndex == -1) {
		return make_float3(0,0,0);
	}
	else {

		objectInfo currObject = scene.objects[hit.objectIndex];
		float3 reflected = make_float3(0, 0, 0);
		float3 refracted = make_float3(0, 0, 0);
		float3 nextPos = hit.pos;
		float3 normal = hit.normal;

		//if (hitTri && length(hitPosTri - currRayPos) < length(nextPos - currRayPos)) {
		//	return make_float3(1, 0, 0);
		//}

		float extraReflection = 0;
		float3 extraColor;
		float3 bias = 0.001 * normal;
		float extraColorSize = 0; 
		if (currObject.refractivity > 0.) {
			float kr;
			bool outside = dot(currRayDir, normal) < 0;
			fresnel(currRayDir, normal, outside? currObject.refractiveIndex : 1 / currObject.refractiveIndex, kr);


			if (kr <= 1) {
				extraColorSize = outside ? 0 : min(1-kr, length(nextPos - currRayPos) * currObject.insideColorDensity);
				float3 refractionDirection = normalize(refract(currRayDir, normal, currObject.refractiveIndex));
				float3 refractionRayOrig = outside ? nextPos - bias : nextPos + bias;

				refracted = currObject.refractivity *(1-kr - extraColorSize)* trace(refractionRayOrig, refractionDirection, remainingDepth - 1, scene);
			}
			extraReflection = min(1.,kr - extraColorSize) * currObject.refractivity;

		}
		if (currObject.reflectivity + extraReflection > 0.) {
			float3 reflectDir = reflect(currRayDir, normal);
			reflected = (currObject.reflectivity + extraReflection )* trace(nextPos + bias, reflectDir, remainingDepth - 1, scene);
		}
		float3 color = (1 - currObject.reflectivity - extraReflection - currObject.refractivity + extraColorSize) * currObject.color;
		return 1000 * (1 / powf(length(nextPos - LIGHT_POS), 2)) /** getShadowTerm(nextPos + bias, scene) */* color + reflected + refracted;
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

	//sceneInfo info = 

	float3 out = 255*trace(firstPlanePos, dirVector, 5, pointers.scene/*currTime, pointers.objects, pointers.numObjects, pointers.meshes, pointers.numMeshes*/);


	//float3 out = 50*pointers.scene.meshes[0].vertices[10];
	//out = 128*make_float3(pointers.scene.meshes
	pointers.g_odata[y * imgw + x] = rgbToInt(out.x, out.y, out.z);
}
extern "C" void
launch_cudaRender(dim3 grid, dim3 block, int sbytes, inputPointers pointers, int imgw, int imgh, float currTime,inputStruct input)
{

	cudaRender << < grid, block, sbytes >> >(pointers, imgw, imgh, currTime, input);
}

