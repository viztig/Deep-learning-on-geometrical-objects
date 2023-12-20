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

/*
a Point structure to store the 3-d coordinates x,y,z
*/
struct Point
{
    double x, y, z;
    Point() : x(0.0), y(0.0), z(0.0) {}
    Point(const double &_x, const double &_y, const double &_z) : x(_x), y(_y), z(_z) {}
};
/*
set of comapre functions to sort the matrix in that particular order of coordinate axis
Eg: if you have to sort the coordinate axis with resprect to the x axis then:std::sort(v.begin(), v.end(), compareZ);
where v=vector<Point>: a vector which store Point
*/
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
/*
read_step to read the .step file with path as filepath and convert it into point cloud .ply file with path as outfilepath
*/
std::string read_step(std::string filepath, std::string outfilepath)
{
    std::ifstream infile(filepath);
    std::ofstream plyFile(outfilepath);
    std::string line;
    int i = 0;
    int k = 0;
    std::vector<Point> v;
    while (std::getline(infile, line))
    {
        if (line[0] == '#')
        {
            int line_i = 0;
            while (line_i < line.size())
            {
                if (line[line_i] == 'C')
                {
                    break;
                }
                else
                {
                    line_i++;
                }
            }
            if (line[line_i + 14] == 'T' && line_i + 21 < line.size())
            {
                std::string sub_line = line.substr(line_i + 21, line.size());
                std::istringstream iss(sub_line);
                std::string part;
                // if (std::getline(iss, part, ' '));
                int c = 0;
                std::vector<double> a(3);
                while (getline(iss, part, ',') && c < 3)
                {
                    double num = 0.0;
                    try
                    {
                        num = std::stod(part);
                    }
                    catch (const std::invalid_argument &ex)
                    {
                        std::cerr << "Caught std::invalid_argument: " << ex.what() << std::endl;
                    }
                    a[c] = num;
                    c++;
                }
                if (a[0] != 0 || a[1] != 0 || a[2] != 0)
                {
                    if (a[1] > 0)
                    {
                        a[1] *= -1;
                    }
                    Point p(a[0], a[1], a[2]);
                    v.push_back(p);

                    plyFile << p.x space p.y space p.z << std ::endl;
                }
            }
        }
    }
    infile.close();
    plyFile.close();
    std::ofstream new_plyFile(outfilepath);
    new_plyFile << "ply" << std::endl;
    new_plyFile << "format ascii 1.0" << std::endl;
    new_plyFile << "element vertex " << v.size() - 1 << std::endl;
    new_plyFile << "property float x" << std::endl;
    new_plyFile << "property float y" << std::endl;
    new_plyFile << "property float z" << std::endl;
    new_plyFile << "end_header" << std::endl;
    new_plyFile.close();
    return outfilepath;
}
/*
read_obj to read the .obj file with path as filepath and convert it into point cloud .ply file with path as outfilepath
*/
std::string read_obj(std::string filepath, std::string outfilepath)
{
    std::ifstream infile(filepath);
    std::ofstream plyFile(outfilepath);
    std::string line;
    int i = 0;
    std::vector<Point> v;
    while (std::getline(infile, line))
    {
        if (line[0] == 'v' && line[1] != 'n')
        {
            std::istringstream iss(line);
            std::string part;
            if (std::getline(iss, part, ' '))
                ;
            int c = 0;
            std::vector<double> a(3);
            while (getline(iss, part, ' ') && c < 3)
            {
                double num = 0.0;
                try
                {
                    num = std::stod(part);
                }
                catch (const std::invalid_argument &ex)
                {
                    std::cerr << "Caught std::invalid_argument: " << ex.what() << std::endl;
                }
                a[c] = num;
                c++;
            }
            if (a[0] != 0 || a[1] != 0 || a[2] != 0)
            {
                if (a[1] > 0)
                {
                    a[1] *= -1;
                }
                Point p(a[0], a[1], a[2]);
                v.push_back(p);

                plyFile << p.x space p.y space p.z << std ::endl;
            }
        }
    }
    infile.close();
    plyFile.close();
    std::ofstream new_plyFile(outfilepath);
    new_plyFile << "ply" << std::endl;
    new_plyFile << "format ascii 1.0" << std::endl;
    new_plyFile << "element vertex " << v.size() - 1 << std::endl;
    new_plyFile << "property float x" << std::endl;
    new_plyFile << "property float y" << std::endl;
    new_plyFile << "property float z" << std::endl;
    new_plyFile << "end_header" << std::endl;
    new_plyFile.close();
    return outfilepath;
}
std::string readFile_key(std::string filepath, std::string outfilepath)
{
    std::ofstream plyFile(outfilepath);
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
            if (line == "$")
            {
                break;
            }
        }
    }
    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << i - 1 << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    plyFile << "end_header" << std::endl;
    infile.close();
    std::ifstream newinfile(filepath);
    int j = 0;
    while (std::getline(newinfile, line))
    {

        if (j >= first)
        {
            std::istringstream iss(line);
            double fnum;
            iss >> fnum;
            double num;
            int c = 0;
            while (iss >> num && c < 3)
            {
                plyFile << num << " ";
                c++;
            }
            iss >> num;
            plyFile << "\n";
            if (line == "$")
            {
                break;
            }
        }
        else
        {
            j++;
        }
    }

    newinfile.close();
    plyFile.close();
    return outfilepath;
}

std::string readFile_inp(std::string filepath, std::string outfilepath)
{
    std::ofstream plyFile(outfilepath);
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
    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << i - 1 << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    plyFile << "end_header" << std::endl;
    infile.close();
    std::ifstream newinfile(filepath);
    int j = 0;
    while (std::getline(newinfile, line))
    {

        if (j >= first)
        {
            std::istringstream iss(line);
            std::string part;
            if (std::getline(iss, part, ','))
                ;
            int c = 0;
            while (getline(iss, part, ',') && c < 3)
            {
                double num = std::stod(part);
                plyFile << num << " ";
                c++;
            }
            plyFile << "\n";
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

    newinfile.close();
    plyFile.close();
    return outfilepath;
}
std::string get_file_extension(std::string file_path)
{
    int Size = 0;
    while (file_path[Size] != '\0')
        Size++;
    std::string s;
    int size = 0;
    for (int i = Size; i >= 0; i--)
    {
        if (file_path[i] == '.')
        {
            size = i;
            break;
        }
    }
    for (int i = size; i < Size; i++)
    {
        s += file_path[i];
    }
    return s;
}
void read_directory(const char *path, const char *folder_path)
{
    DIR *dir;
    struct dirent *ent;
    std::string pth = path;
    std::string folderpath = folder_path;
    if ((dir = opendir(path)) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            if (ent->d_name[0] != '.')
            {
                std::string filepath = ent->d_name;
                std::string input_filepath = pth + '/' + filepath;
                std::string s = get_file_extension(input_filepath);
                std::string file;
                if (s == ".step")
                {
                    std::string output_dirpath = folderpath + '/' + filepath.substr(0, filepath.size() - 5);
                    std::string output_filepath = output_dirpath + ".ply";
                    file = read_step(input_filepath, output_filepath);
                }
                if (s == ".obj")
                {
                    std::string output_dirpath = folderpath + '/' + filepath.substr(0, filepath.size() - 4);
                    std::string output_filepath = output_dirpath + ".ply";
                    file = read_step(input_filepath, output_filepath);
                }
                if (s == ".key")
                {
                    std::string output_dirpath = folderpath + '/' + filepath.substr(0, filepath.size() - 4);
                    std::string output_filepath = output_dirpath + ".ply";
                    file = read_step(input_filepath, output_filepath);
                }
                if (s == ".inp" || s == ".inc")
                {
                    std::string output_dirpath = folderpath + '/' + filepath.substr(0, filepath.size() - 4);
                    std::string output_filepath = output_dirpath + ".ply";
                    file = read_step(input_filepath, output_filepath);
                }
            }
        }
    }
}
/*
std::vector<Point> read_dir(const char *path,std::string mode)
{
    std::vector<Point> all_v;
    DIR *dir;
    struct dirent *ent;
    std::string _path = path;
    if ((dir = opendir(path)) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            if (ent->d_type == DT_REG)
            {
                std::string file_name = ent->d_name;
                std::string ext = file_name.substr(file_name.size() - 3, file_name.size());
                // std::cout << ext << std::endl;
                if (ext == ".jt")
                {
                    std::string full_path = _path + file_name;
                    // std::cout << jt_path << std::endl;
                    if (mode=="step"){std::vector<Point> v = read_step(full_path);}
                    std::cout << v.size() << std::endl;
                    for (Point &x : v)
                    {
                        all_v.push_back(x);
                    }
                }
            }
        }
    }
    return all_v;
}
*/
std::vector<Point> read_vector(std::string filepath)
{
    std::vector<Point> v;
    std::ifstream infile(filepath);
    std::string line;
    int first = 6;
    int j = 0;
    while (std::getline(infile, line))
    {
        if (j > first)
        {
            std::istringstream iss(line);
            std::string part;
            int c = 0;
            std::vector<double> a;
            while (std::getline(iss, part, ' '))
            {
                try
                {
                    a.push_back(std::stod(part));
                }
                catch (const std::invalid_argument &ex)
                {
                    std::cerr << "Caught std::invalid_argument: " << ex.what() << std::endl;
                }
            }
            Point p;
            p.x = a[0];
            p.y = a[1];
            p.z = a[2];
            v.push_back(p);
        }
        else
        {
            j++;
        }
    }
    return v;
}
void getParam(std::vector<Point> &v)
{
    std::sort(v.begin(), v.end(), compareZ);
}
int main(int, char *argv[])
{
    std::string s = get_file_extension(argv[1]);
    std::string file;
    if (s == ".step")
    {
        file = read_step(argv[1], argv[2]);
    }
    if (s == ".obj")
    {
        file = read_obj(argv[1], argv[2]);
    }
    if (s == ".key")
    {
        file = readFile_key(argv[1], argv[2]);
    }
    if (s == ".inp" || s == ".inc")
    {
        file = readFile_inp(argv[1], argv[2]);
    }
    // std::vector<Point> v = read_step(argv[1], argv[2]);
    //   std::vector<Point> v = read_dir(argv[1]);
    // std::cout << "vector size:" space v.size() space v[0].x space v[v.size() - 2].z << std::endl;
    //   std::string s = save_ply(argv[2], v);
    //  std::vector<Point> v = read_vector(argv[1]);
    //  getParam(v);
    //  std::cout << v[0].y space v[1].y << std::endl;
    return 0;
}
