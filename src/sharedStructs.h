#pragma once

#include "cuda_runtime.h"

#define AIR_DENSITY 0.001
#define AIR_COLOR 1.0*make_float3(53.0/255, 81.0/255, 98.0/255)
#define WATER_COLOR make_float3(0,0.0,0.1)
#define WATER_DENSITY 0.07


#define WIDTH 1920
#define HEIGHT 1080

#define LIGHT_BUFFER_WORLD_SIZE 200
#define LIGHT_PLANE_SIZE 200
#define LIGHT_BUFFER_WIDTH 1024
#define LIGHT_BUFFER_THICKNESS 20
#define LIGHT_BUFFER_THICKNESS_SIZE 100


// dont change these
#define LIGHT_BUFFER_WORLD_RATIO (1. / LIGHT_BUFFER_WORLD_SIZE)
#define LIGHT_BUFFER_THICKNESS_WORLD_RATIO (1. / LIGHT_BUFFER_THICKNESS_SIZE)


// total size will be pow(GRID_SIZE,3) bc of xyz
#define GRID_SIZE 15
#define GRID_SIZE2 GRID_SIZE*GRID_SIZE

#define GRID_POS(x,y,z) GRID_SIZE2*x + GRID_SIZE*y + z


// global grid is currently unused
#define GLOBAL_GRID_SIZE 4
#define GLOBAL_GRID_SIZE2 GLOBAL_GRID_SIZE*GLOBAL_GRID_SIZE
#define GLOBAL_GRID_MAX make_float3(100,100,100)
#define GLOBAL_GRID_MIN make_float3(-100,-100,-100)
#define GLOBAL_GRID_DIMENSIONS (GLOBAL_GRID_MAX - GLOBAL_GRID_MIN) *(1./GLOBAL_GRID_SIZE)

#define GLOBAL_GRID_POS(x,y,z) GLOBAL_GRID_SIZE2*x + GLOBAL_GRID_SIZE*y + z



// BBM stuff
#define DEFAULT_BBM_SIDE_RES 16
#define DEFAULT_BBM_ANGLE_RES 15


#define PI 3.141592654f
#define DEG(rad) rad*57.2957795
#define RAD(deg) deg/57.2957795



inline __device__ float3 operator+(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator+(const float3& a, const float& b) {
	return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __device__ float3 operator*(const float3& a, const float3& b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ float3 operator/(const float3& a, const float3& b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}


inline __device__ float3 operator*(const float& a, const float3& b) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __device__ float3 operator*(const float3& b, const float& a) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __device__ float3 operator-(const float3& a, const float3& b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 fromShort(const short3& a) {
	return make_float3(a.x, a.y, a.z);
}




// shorties
inline __device__ short3 operator+(const short3& a, const short3& b) {
	return make_short3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ short3 operator+(const short3& a, const short& b) {
	return make_short3(a.x + b, a.y + b, a.z + b);
}

inline __device__ short3 operator*(const short3& a, const short3& b) {
	return make_short3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ short3 operator/(const short3& a, const short3& b) {
	return make_short3(a.x / b.x, a.y / b.y, a.z / b.z);
}


inline __device__ short3 operator*(const short& a, const short3& b) {
	return make_short3(a * b.x, a * b.y, a * b.z);
}

inline __device__ short3 operator*(const short3& b, const short& a) {
	return make_short3(a * b.x, a * b.y, a * b.z);
}

inline __device__ short3 operator-(const short3& a, const short3& b) {
	return make_short3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ short3 fromLong(const float3& a) {
	return make_short3(a.x, a.y, a.z);
}





inline __device__ float3 floor(const float3& a) {
	return make_float3(floor(a.x), floor(a.y), floor(a.z));
}

inline __device__ float2 floor(const float2& a) {
	return make_float2(floor(a.x), floor(a.y));
}

inline __device__ float2 operator*(const float& a, const float2& b) {
	return make_float2(a * b.x, a * b.y);
}

inline __device__ float2 operator+(const float2& a, const float2& b) {
	return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ float2 operator*(const float2& a, const float2& b) {
	return make_float2(a.x * b.x, a.y * b.y);
}


inline __device__  float dot(float3 v1, float3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline __device__  float dot(float2 v1, float2 v2)
{
	return v1.x * v2.x + v1.y * v2.y;
}

inline __device__  float3 cross(float3 v1, float3 v2)
{
	return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

inline __device__ float length(float3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline __device__ float length1(float3 v)
{
	return v.x + v.y + v.z;
}

inline __device__ float3 inverse(float3 v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

inline __device__ float3 normalize(float3 v)
{
	float invLen = 1 / sqrtf(dot(v, v));
	return invLen * v;
}




struct shapeInfo {
	float3 pos = float3();
	float3 normal = float3();
	float rad = 0.;
};

struct sphereInfo {
	float3 pos = float3();
	float rad = 0.;
	float rad2 = 0.;
};

inline __device__ shapeInfo make_shapeInfo(float3 pos, float3 normal, float rad) {
	shapeInfo info;
	info.pos = pos;
	info.normal = normal;
	info.rad = rad;
	return info;
}

inline __device__ sphereInfo make_sphereInfo(float3 pos, float rad) {
	sphereInfo s;
	s.pos = pos;
	s.rad = rad;
	s.rad2 = rad * rad;
	return s;
}


struct planeInfo {
	float3 point;
	float3 normal;
};

inline __device__ planeInfo make_planeInfo(float3 point, float3 normal) {
	planeInfo p;
	p.point = point;
	p.normal = normal;
	return p;
}

enum shape { sphere, plane, water };

struct rayHitInfo {
	float reflectivity = 0.;
	float refractivity = 0.;
	float refractiveIndex = 1.0;
	float insideColorDensity = 0.;
	float3 color = float3{ 0,0,0 };
	float roughness = 0.;
};


inline __device__ rayHitInfo make_rayHitInfo(float inReflectivity, float inRefractivity, float inRefractiveIndex, float inInsideColorDensity, float3 inColor, float roughness) {
	rayHitInfo r;
	r.reflectivity = inReflectivity;
	r.refractivity = inRefractivity;
	r.refractiveIndex = inRefractiveIndex;
	r.insideColorDensity = inInsideColorDensity;
	r.color = inColor;
	r.roughness = roughness;
	return r;
}

struct objectInfo {
	shape s = shape();
	shapeInfo shapeData = shapeInfo();
	rayHitInfo rayInfo = rayHitInfo();

};

inline __device__ objectInfo make_objectInfo(shape s, shapeInfo shapeData, float reflectivity, float3 color, float refractivity, float refractiveIndex, float insideColorDensity, float roughness) {
	objectInfo o;
	o.s = s;
	o.shapeData = shapeData;

	o.rayInfo = make_rayHitInfo(reflectivity, refractivity, refractiveIndex, insideColorDensity, color, roughness);
	return o;
}



struct triangleMesh {
	float3* vertices;
	float3* normals;
	//float3* tangents; 
	float2* UVs;
	unsigned int* indices;
	int numIndices = 0;
	int numVertices = 0;

	rayHitInfo rayInfo;

	// texture
	unsigned int* diffuseMap;
	unsigned int* normalMap;

	// acceleration structure
	float3 bbMin = float3();
	float3 bbMax = float3();
	float rad = 0.;
	unsigned int** grid = nullptr; // lists with unsigned int marking which triangles intersect
	unsigned int* gridSizes = nullptr;
	float3 gridBoxDimensions = float3{ 0,0,0 };
};


struct BBMRes {
	float hitRatio = 0.f;
	float3 startP = float3();
	float3 colorOut = make_float3(0,0,0);
	float3 startPNormal = float3();

	short ray1Power = 0.f;
	float3 ray1Orig = float3();
	float3 ray1Dir = float3();
};

inline __device__ BBMRes operator+(const BBMRes& a, const BBMRes& b) {
	#define ADDBBMITEM(Name) toReturn . Name = a . Name + b . Name ;
	BBMRes toReturn;
	ADDBBMITEM(hitRatio)
	ADDBBMITEM(startP)
	ADDBBMITEM(colorOut)
	ADDBBMITEM(startPNormal)
	ADDBBMITEM(ray1Power)
	ADDBBMITEM(ray1Orig)
	ADDBBMITEM(ray1Dir)
	return toReturn;
}
inline __device__ BBMRes operator*(const BBMRes& a, const float& b) {
	#define MULBBMItem(Name) toReturn . Name = a . Name * b ;
	BBMRes toReturn;
	MULBBMItem(hitRatio)
	MULBBMItem(startP)
	MULBBMItem(colorOut)
	MULBBMItem(startPNormal)
	MULBBMItem(ray1Power)
	MULBBMItem(ray1Orig)
	MULBBMItem(ray1Dir)
	return toReturn;
}





struct blackBoxMesh {
	BBMRes* texture = nullptr;
	float3 bbMin = make_float3(0, 0, 0);
	float3 bbMax = make_float3(0,0,0);
	//float rad;

	int sideResolution = DEFAULT_BBM_SIDE_RES;
	int angleResolution = DEFAULT_BBM_ANGLE_RES;
};

struct BBMPassInput {
	blackBoxMesh bbm = blackBoxMesh();
	triangleMesh mesh = triangleMesh();
};


struct sceneInfo {

	// unused stuff for global grid
	unsigned int** gridMeshes;
	unsigned int** gridObjects;
	unsigned int* gridMeshesSizes;
	unsigned int* gridObjectsSizes;

	// objects are pure mathematical objects, while meshes are triangle meshes
	float currTime;
	objectInfo* objects;
	int numObjects;

	triangleMesh* meshes;
	int numMeshes;

	blackBoxMesh* bbMeshes;
	int numBBMeshes = 0;

};

struct PostProcessPointers {
	unsigned int* inputImage;
	unsigned int* processRead;
	unsigned int* processWrite;
	unsigned int* finalOut;
};

struct inputImage {
	unsigned char* image = nullptr;
	unsigned int width = 0;
	unsigned int height = 0;
};


struct inputPointers {
	unsigned int* image1; // texture position
	unsigned int* lightImage; // light texture position

	inputImage waterNormal;
	sceneInfo scene;

};


struct hitInfo {
	rayHitInfo info;
	bool hit = false;
	float3 pos;
	float3 normal;

	bool normalIsInversed = false;

	BBMRes bbmHit;
	//bool isBBM = false;
};

struct prevHitInfo {
	float insideColorDensity = 0.0;
	float3 color;
	float refractiveIndex = 1.0;
};


inline __device__ prevHitInfo make_prevHitInfo(const hitInfo& info) {
	prevHitInfo toReturn;
	toReturn.insideColorDensity = info.info.insideColorDensity;
	toReturn.color = info.info.color;
	toReturn.refractiveIndex = info.info.refractiveIndex;
	return toReturn;

}


struct inputStruct {
	float currPosX = 0.f;
	float currPosY = 5.f;
	float currPosZ = 10.f;

	float forwardX = 0.f;
	float forwardY = 0.f;
	float forwardZ = 0.f;

	float upX = 0.f;
	float upY = 0.f;
	float upZ = 0.f;

	prevHitInfo beginMedium = prevHitInfo();
};


#define LERP_BBM(name) toReturn . name = b1 . name + ((b2 . name ) - ( b1. name )) * fac
inline __device__ BBMRes LerpBBM(const BBMRes b1, const BBMRes b2, const float fac) {
	BBMRes toReturn;
	LERP_BBM(hitRatio);
	LERP_BBM(startP);
	LERP_BBM(colorOut);
	if (b1.hitRatio < 0.1) {toReturn.colorOut = b2.colorOut; }else if (b2.hitRatio < 0.1) { toReturn.colorOut = b1.colorOut; }
	LERP_BBM(startPNormal);
	if (b1.hitRatio < 0.1) {toReturn.startPNormal = b2.startPNormal; }else if (b2.hitRatio < 0.1) { toReturn.startPNormal = b1.startPNormal; }

	//LERP_BBM(ray1Power);
	//LERP_BBM(ray1Orig);
	//LERP_BBM(ray1Dir);

	//if (b1.hit ^ b2.hit) {
	//toReturn.ray1Power = (!b1.hit ? 1 : 0)+ ((!b2.hit ? 1 : 0)- (!b1.hit ? 1 : 0)) * fac;
	//}

	return toReturn;
}

inline __device__ bool intersectBox(const float3 orig, const float3 dir, const float3 min, const float3 max, float& tmin, float& tmax)
{
	tmin = (min.x - orig.x) / dir.x;
	tmax = (max.x - orig.x) / dir.x;

	if (tmin > tmax) {
		float temp = tmin; tmin = tmax; tmax = temp;
	}

	float tymin = (min.y - orig.y) / dir.y;
	float tymax = (max.y - orig.y) / dir.y;

	if (tymin > tymax) {
		float temp = tymin; tymin = tymax; tymax = temp;
	}

	if ((tmin > tymax) || (tymin > tmax))
		return false;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	float tzmin = (min.z - orig.z) / dir.z;
	float tzmax = (max.z - orig.z) / dir.z;

	if (tzmin > tzmax) {
		float temp = tzmin; tzmin = tzmax; tzmax = temp;

	}

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;

	return true;


}


inline __device__ bool intersectsSphere(const float3& origin, const float3& dir, const float3 pos, const float rad, float& t) {

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
inline __device__ bool intersectPlane(const shapeInfo& p, const float3& l0, const float3& l, float& t)
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

inline  __device__ bool RayIntersectsTriangle(float3 rayOrigin,
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
	if (v < 0.0 || (u + v) > 1.0)
		return false;
	// At this stage we can compute t to find out where the intersection point is on the line.
	t = f * dot(edge2, q);

	return t > EPSILON && !((u < 0.0 || u > 1.0) || (v < 0.0 || ((u + v) > 1.0)));
}

inline __device__ float3 sphericalCoordsToRectangular(const float3 sphere) {
	return make_float3(sphere.x * sin(sphere.z) * cos(sphere.y), sphere.x * sin(sphere.z) * sin(sphere.y), sphere.x * cos(sphere.z));
}

inline __device__ float3 rectangularCoordsToSpherical(const float3 rect) {

	float r = sqrt(rect.x * rect.x + rect.y * rect.y + rect.z * rect.z);
	float p = atan2(rect.y, rect.x);
	float o = acos(rect.z / r);
	return make_float3(r, p, o);
}

#define MAX(a,b) a < b ? b : a
#define MIN(a,b) a > b ? b : a


inline __device__ float3 getTan(float3 vec) {
	float3 spherical = rectangularCoordsToSpherical(vec);
	return sphericalCoordsToRectangular(spherical + make_float3(0, 0, PI/2));
}




inline __device__ int directionToInt(float3 dir) {
	if (dot(dir, make_float3(1, 0, 0)) > 0.999) {
		return 0;
	}
	else if (dot(dir, make_float3(-1, 0, 0)) > 0.999) {
		return 1;
	}
	else if (dot(dir, make_float3(0, 1, 0)) > 0.999) {
		return 2;
	}
	else if (dot(dir, make_float3(0, -1, 0)) > 0.999) {
		return 3;
	}
	else if (dot(dir, make_float3(0, 0, 1)) > 0.999) {
		return 4;
	}
	else if (dot(dir, make_float3(0, 0, -1)) > 0.999) {
		return 5;
	}
	return  -1000000; // you dun goofed if this happens


}

inline __device__ float3 intToDirection(int directionIndex) {
	float3 direction = make_float3(0,0,0);
	switch (directionIndex) {
	case 0: direction = make_float3(1, 0, 0); break;
	case 1: direction = make_float3(-1, 0, 0); break;
	case 2: direction = make_float3(0, 1, 0); break;
	case 3: direction = make_float3(0, -1, 0); break;
	case 4: direction = make_float3(0, 0, 1); break;
	case 5: direction = make_float3(0, 0, -1); break;
	}
	return direction;
}


inline __device__ void getXYAndAngleCoordinates(const float3 position, const float3 lookingDir, const blackBoxMesh bbm, int &directionIndex, float& xPos, float &yPos, float &angleYPos, float &angleZPos) {
	float3 center = (bbm.bbMax + bbm.bbMin) * 0.5;
	float3 centerToContact = position - center;
	float3 bbMinToContact = position - bbm.bbMin;

	float3 bbmChongos = (bbm.bbMax - bbm.bbMin);
	float3 adjustedCenterToContact = centerToContact / bbmChongos;

	float3 majorDir;
	if ((abs(adjustedCenterToContact.x) > abs(adjustedCenterToContact.y)) && (abs(adjustedCenterToContact.x) > abs(adjustedCenterToContact.z))) {
		majorDir = make_float3(centerToContact.x, 0, 0);
	}
	else if (abs(adjustedCenterToContact.y) > abs(adjustedCenterToContact.z)) {
		majorDir = make_float3(0, centerToContact.y, 0);
	}
	else {
		majorDir = make_float3(0, 0, centerToContact.z);
	}


	majorDir = /*make_float3(0, 0, 1);//*/ (majorDir * (1 / length(majorDir))); // normalize length
	float3 tan = /*make_float3(0, 1, 0);//*/ getTan(majorDir);
	float3 biTan = /*make_float3(1, 0, 0);//*/ cross(majorDir, tan);
	directionIndex = directionToInt(majorDir);

	float distT = dot(tan, bbMinToContact);
	float distB = dot(biTan, bbMinToContact);

	float maxDistT = dot(tan, bbmChongos);
	float maxDistB = dot(biTan, bbmChongos);

	xPos = (distT / maxDistT)* bbm.sideResolution - 0.5f;
	yPos = (distB / maxDistB) * bbm.sideResolution - 0.5f;
	xPos = MIN(MAX(xPos, 0.0f), (float) bbm.sideResolution - 1);
	yPos = MIN(MAX(yPos, 0.0f), (float) bbm.sideResolution - 1);


	// angle
	float prelAngleYPos = dot(lookingDir, tan);
	float prelAngleZPos = dot(lookingDir, biTan);


	float stepLen = /*PI / 2 / bbm.angleResolution + 1;//*/ PI / ((bbm.angleResolution + 1));
	float lowestP = /*-PI / 4;//*/ -stepLen * ((bbm.angleResolution - 1) / 2);

	angleYPos = (asin(prelAngleYPos) - lowestP)/stepLen;
	angleZPos = (asin(prelAngleZPos) - lowestP)/stepLen;



}

inline __device__ int getIndex(int directionIndex, int XIndex, int YIndex, int angleYIndex, int angleZIndex, const blackBoxMesh bbm) {
	int XYIndex = YIndex* bbm.sideResolution + XIndex;
	int angleIndex = angleYIndex * bbm.angleResolution + angleZIndex;

	return directionIndex * (bbm.sideResolution * bbm.sideResolution) * (bbm.angleResolution * bbm.angleResolution) + XYIndex * (bbm.angleResolution * bbm.angleResolution) + angleIndex;
}

inline __device__ BBMRes rectangularCoordsToLerpedAngleValue(int directionIndex, float xPos, float yPos, float yAnglePos, float zAnglePos, const blackBoxMesh bbm) {
	int angleStepsY = MAX(0, floor(yAnglePos));
	int angleStepsZ = MAX(0, floor(zAnglePos));
	float angleStepsRatY = MIN(MAX(0, yAnglePos - angleStepsY), 1.0f);
	float angleStepsRatZ = MIN(MAX(0, zAnglePos - angleStepsZ), 1.0f);


	int XSteps = MAX(0, MIN(round(xPos), bbm.sideResolution - 1));
	int YSteps = MAX(0, MIN(round(yPos), bbm.sideResolution - 1));

	int nextAngleY = MIN(angleStepsY + 1, bbm.angleResolution - 1);
	int nextAngleZ = MIN(angleStepsZ + 1, bbm.angleResolution - 1);


	BBMRes UL = bbm.texture[getIndex(directionIndex, XSteps, YSteps, angleStepsY, angleStepsZ, bbm)];
	BBMRes UR = bbm.texture[getIndex(directionIndex, XSteps, YSteps, nextAngleY, angleStepsZ, bbm)];
	BBMRes LL = bbm.texture[getIndex(directionIndex, XSteps, YSteps, angleStepsY, nextAngleZ, bbm)];
	BBMRes LR = bbm.texture[getIndex(directionIndex, XSteps, YSteps, nextAngleY, nextAngleZ, bbm)];

	BBMRes combinedUpper = LerpBBM(UL, UR, angleStepsRatY);
	BBMRes combinedDown = LerpBBM(LL, LR, angleStepsRatY);

	BBMRes combined = LerpBBM(combinedUpper, combinedDown, angleStepsRatZ);
	return combined;
}



inline __device__ BBMRes rectangularCoordsToLerpedValue(const float3 position, const float3 lookingDir, const blackBoxMesh bbm) {
	int directionIndex;
	float xPos, yPos, yAnglePos, zAnglePos;

	getXYAndAngleCoordinates(position, lookingDir, bbm, directionIndex, xPos, yPos, yAnglePos, zAnglePos);
	//xPos += 0.5f;
	//yPos += 0.5f;

	int floorX = floor(xPos);
	int floorY = floor(yPos);

	float xRat = xPos - floorX;
	float yRat = yPos - floorY;


	BBMRes LL = rectangularCoordsToLerpedAngleValue(directionIndex, floorX, floorY, yAnglePos, zAnglePos, bbm);
	BBMRes LR = rectangularCoordsToLerpedAngleValue(directionIndex, floorX +1, floorY, yAnglePos, zAnglePos, bbm);
	BBMRes UL = rectangularCoordsToLerpedAngleValue(directionIndex, floorX, floorY +1, yAnglePos, zAnglePos, bbm);
	BBMRes UR = rectangularCoordsToLerpedAngleValue(directionIndex, floorX +1, floorY +1, yAnglePos, zAnglePos, bbm);

	//if (xPos < 0) {
	//	LL = BBMRes();
	//	UL = BBMRes();
	//}
	//if (yPos < 0) {
	//	LL = BBMRes();
	//	LR = BBMRes();
	//}

	BBMRes combinedUpper = LerpBBM(UL, UR, xRat);
	BBMRes combinedDown = LerpBBM(LL, LR, xRat);

	BBMRes combined = LerpBBM(combinedDown, combinedUpper, yRat);
	return combined;
}




inline __device__ int rectangularCoordsToIndex(const float3 position, const float3 lookingDir, const blackBoxMesh bbm) {


	int directionIndex;
	float xPos, yPos, yAnglePos, zAnglePos;
	getXYAndAngleCoordinates(position, lookingDir, bbm, directionIndex, xPos, yPos, yAnglePos, zAnglePos);

	int XIndex = (int) round(xPos); 
	int YIndex = (int) round(yPos);

	XIndex = MAX(MIN(XIndex, bbm.sideResolution - 1),0);
	YIndex = MAX(MIN(YIndex, bbm.sideResolution - 1),0);
	//int XYIndex = YIndex * bbm.sideResolution + XIndex;


	int angleStepsY = /*(bbm.angleResolution - 1) / 2;//*/ MIN(MAX(round(yAnglePos),0),(bbm.angleResolution-1));
	int angleStepsZ = /*(bbm.angleResolution - 1) / 2;//*/ MIN(MAX(round(zAnglePos), 0), (bbm.angleResolution - 1));

	//int angleIndex =  angleStepsY* bbm.angleResolution + angleStepsZ;

	return getIndex(directionIndex, XIndex, YIndex, angleStepsY, angleStepsZ, bbm);
	//return directionIndex * (bbm.sideResolution*bbm.sideResolution)* (bbm.angleResolution * bbm.angleResolution) + XYIndex * (bbm.angleResolution * bbm.angleResolution) + angleIndex;
}

