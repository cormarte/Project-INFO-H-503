float cosrz, sinrz;
__sincosf(-rz * PI / 180.0f, &sinrz, &cosrz);

int centerX = width / 2 - tx;
int centerY = height / 2 - ty;

int originalX = (int)((x - centerX)*cosrz - (y - centerY)*sinrz - tx + centerX);
int originalY = (int)((x - centerX)*sinrz + (y - centerY)*cosrz - ty + centerY);