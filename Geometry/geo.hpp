#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#define space << " " <<

class Point
{
public:
    double x;
    double y;
    double z;
    Point() : x(0.0), y(0.0), z(0.0) {}
    Point(const double &_x, const double &_y, const double &_z) : x(_x), y(_y), z(_z) {}
    ~Point() {}
    double gradX(Point &, Point &);
    double gradY(Point &, Point &);
    double gradZ(Point &, Point &);
    // static bool compareX(Point &, Point &);
    // static bool compareY(Point &, Point &);
    // static bool compareZ(Point &, Point &);
    static std::vector<Point> getOutlineZ(std::string filepath);
    static std::vector<Point> getOutlineZY(std::string filepath);
    static std::vector<Point> getOutlineZX(std::string filepath);
};

double Point::gradX(Point &a, Point &b)
{
    return double((a.z - b.z) / (a.y - b.y));
}
double Point::gradY(Point &a, Point &b)
{
    return double((a.z - b.z) / (a.x - b.x));
}
double Point::gradZ(Point &a, Point &b)
{
    return double((a.x - b.x) / (a.y - b.y));
}
bool compareX(Point &a, Point &b)
{
    return a.x > b.x;
}
bool compareY(Point &a, Point &b)
{
    return a.y > b.y;
}
bool compareZ(Point &a, Point &b)
{
    return a.z > b.z;
}   
bool compareXZ(Point &a, Point &b)
{
    return (a.z > b.z && a.x > b.x);
}
std::vector<Point> readFile_inp(std::string filepath, std::unordered_map<int, int> &mapx, std::unordered_map<int, int> &mapy, std::unordered_map<int, int> &mapz)
{
    std::ifstream infile(filepath);
    std::string line;
    int i = 0;
    int first = 0;
    while (std::getline(infile, line))
    {
        if (i == 0)
        {
            first++;
            if (line == "*NODE")
            {
                i = 1;
            }
        }
        else
        {
            i++;
            if (line[0] == '*')
            {
                break;
            }
        }
    }
    std::cout << "storing vector" << std::endl;
    std::vector<Point> v(i - 1);

    std::ifstream newinfile(filepath);
    int j = 0;
    int k = 0;
    while (std::getline(newinfile, line))
    {
        if (j >= first)
        {
            std::istringstream iss(line);
            std::string part;
            if (std::getline(iss, part, ','))
                ;
            int c = 0;
            std::vector<double> a(3);
            while (getline(iss, part, ',') && c < 3)
            {
                double num = std::stod(part);
                a[c] = num;
                c++;
            }
            v[k].x = a[0];
            v[k].y = a[1];
            v[k].z = a[2];
            if (mapx.count(int(a[0])) == 0)
            {
                mapx[int(a[0])] = 1;
            }
            if (mapy.count(int(a[1])) == 0)
            {
                mapy[int(a[1])] = 1;
            }
            if (mapz.count(int(a[2])) == 0)
            {
                mapz[int(a[2])] = 1;
            }
            k++;
            if (line[0] == '*')
            {
                break;
            }
        }
        else
        {
            j++;
        }
    }
    std::cout << "sorting the vector" << std::endl;
    newinfile.close();
    return v;
}
std::vector<Point> Point::getOutlineZ(std::string filepath)
{
    std::unordered_map<int, int> mapx;
    std::unordered_map<int, int> mapy;
    std::unordered_map<int, int> mapz;
    std::vector<Point> v = readFile_inp(filepath, mapx, mapy, mapz);
    std::sort(v.begin(), v.end(), compareZ);
    std::vector<Point> uniq;
    uniq.reserve(mapz.size());
    for (Point &a : v)
    {
        if (mapz[int(a.z)] == 1)
        {
            Point p(a.x, a.y, a.z);
            uniq.emplace_back(p);
            mapz[int(a.z)] = 0;
        }
    }
    return uniq;
}
std::vector<Point> Point::getOutlineZY(std::string filepath)
{
    std::unordered_map<int, int> mapx;
    std::unordered_map<int, int> mapy;
    std::unordered_map<int, int> mapz;
    std::vector<Point> v = readFile_inp(filepath, mapx, mapy, mapz);
    std::sort(v.begin(), v.end(), compareY);
    std::vector<Point> uniq;
    uniq.reserve(mapz.size());
    for (Point &a : v)
    {
        if (mapz[int(a.z)] == 1)
        {
            Point p(0, a.y, a.z);
            uniq.emplace_back(p);
            mapz[int(a.z)] = 0;
        }
    }
    return uniq;
}
std::vector<Point> Point::getOutlineZX(std::string filepath)
{
    std::unordered_map<int, int> mapx;
    std::unordered_map<int, int> mapy;
    std::unordered_map<int, int> mapz;
    std::vector<Point> v = readFile_inp(filepath, mapx, mapy, mapz);
    std::sort(v.begin(), v.end(), compareX);
    std::vector<Point> uniq;
    uniq.reserve(mapz.size());
    for (Point &a : v)
    {
        if (mapz[int(a.z)] == 1)
        {
            Point p(a.x, 0, a.z);
            uniq.emplace_back(p);
            mapz[int(a.z)] = 0;
        }
    }
    return uniq;
}