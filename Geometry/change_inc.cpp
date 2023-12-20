// Author:VAIBHAV VIKAS (VAVIKAS)
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
#include "permutations.hpp"
#define space << " " <<
/*
0:backrest insert
1:cushion insert
2:backrest bolster
3:cushion bolster
*/
std::string density_5 = "0.000133,17.348241,0.023955,17.012855,0.155487,0.000000";
std::string density_4 = "0.000106,17.351514,0.019173,17.017339,0.155498,0.000000";
std::string density_6 = "0.028748,17.013571,0.000160,17.341243,0.000000,0.155529";
std::string density_7 = "0.036917,18.738705,0.000249,18.789000,0.000000,0.145691";
std::string density_8 = "0.042878,15.974742,-0.003889,0.010000,0.055795,0.072584";
std::string density_9 = "0.051700,17.066469,-0.004601,0.010000,0.053754,0.055315";
std::string density_10 = "0.000637,24.144155,0.067812,24.065763,0.120007,0.000000";
std::string change(std::string &s, int a)
{
    int ind = s.find("pu_rg");
    if (ind)
    {
        s[ind + 5] = 9 + '0';
        s[ind + 6] = 5 + '0';
        s[ind + 8] = a / 10 + '0';
        s[ind + 9] = a % 10 + '0';
    }
    return s.substr(0, ind + 10) + "kpa";
}
std::string generate_filename(const std::string &filename, const std::vector<int> &v)
{
    std::string new_filename = filename.substr(0, filename.size() - 4) + '_';
    for (int i = 0; i < v.size(); i++)
    {
        char c = v[i] / 10 + '0';
        new_filename += c;
        char d = v[i] % 10 + '0';
        new_filename += d;
    }

    new_filename += ".inc";
    return new_filename;
}
void modify_inc(std::string infilepath, std::string outfilepath, const std::vector<int> &density)
{
    std::ofstream newFile(outfilepath);
    std::ifstream infile(infilepath);
    std::string line;
    while (std::getline(infile, line))
    {
        if (line.find("***") < 5)
        {
            break;
        }
        if (line[0] == '*')
        {
            if (line.find("Backrest_Bolster,") != -1 || line.find("BACKREST_BOLSTER,") != -1 || line.find("Backrest_Bolster_Foam,") != -1 || line.find("BACKREST_BOLSTER_FOAM,") != -1)
            {
                std::string s = change(line, density[2]);
                newFile << s << std::endl;
                continue;
            }
            else if (line.find("Backrest_Insert,") != -1 || line.find("BACKREST_INSERT,") != -1 || line.find("Backrest_Insert_Foam,") != -1 || line.find("BACKREST_INSERT_FOAM,") != -1)
            {
                std::string s = change(line, density[0]);
                newFile << s << std::endl;
                continue;
            }
            else if (line.find("Cushion_Bolster,") != -1 || line.find("CUSHION_BOLSTER,") != -1 || line.find("Cushion_Bolster_Foam,") != -1 || line.find("CUSHION_BOLSTER_FOAM,") != -1)
            {
                std::string s = change(line, density[3]);
                newFile << s << std::endl;
                continue;
            }
            else if (line.find("Cushion_Insert,") != -1 || line.find("CUSHION_INSERT,") != -1 || line.find("Cushion_Insert_Foam,") != -1 || line.find("CUSHION_INSERT_FOAM,") != -1)
            {
                std::string s = change(line, density[1]);
                newFile << s << std::endl;
                continue;
            }
            // else if (line.find("MATERIAL, NAME=") != -1)
            //{
            //   std::string s = change(line, density[1]);
            // newFile << s << std::endl;
            // continue;
            //}
            else
            {
                newFile << line << std::endl;
            }
        }
        else
        {
            newFile << line << std::endl;
        }
    }
    newFile << "*MATERIAL, NAME=pu_rg95_04kpa" << std::endl;
    newFile << "*DENSITY" << std::endl;
    newFile << "6.5000E-11,0.0 " << std::endl;
    newFile << "*HYPERFOAM, N =  2" << std::endl;
    newFile << density_4 << std::endl;

    newFile << "*MATERIAL, NAME=pu_rg95_05kpa" << std::endl;
    newFile << "*DENSITY" << std::endl;
    newFile << "6.5000E-11,0.0 " << std::endl;
    newFile << "*HYPERFOAM, N =  2" << std::endl;
    newFile << density_5 << std::endl;

    newFile << "*MATERIAL, NAME=pu_rg95_06kpa" << std::endl;
    newFile << "*DENSITY" << std::endl;
    newFile << "6.5000E-11,0.0 " << std::endl;
    newFile << "*HYPERFOAM, N =  2" << std::endl;
    newFile << density_6 << std::endl;

    newFile << "*MATERIAL, NAME=pu_rg95_07kpa" << std::endl;
    newFile << "*DENSITY" << std::endl;
    newFile << "6.5000E-11,0.0 " << std::endl;
    newFile << "*HYPERFOAM, N =  2" << std::endl;
    newFile << density_7 << std::endl;

    newFile << "*MATERIAL, NAME=pu_rg95_08kpa" << std::endl;
    newFile << "*DENSITY" << std::endl;
    newFile << "6.5000E-11,0.0 " << std::endl;
    newFile << "*HYPERFOAM, N =  2" << std::endl;
    newFile << density_8 << std::endl;

    newFile << "*MATERIAL, NAME=pu_rg95_09kpa" << std::endl;
    newFile << "*DENSITY" << std::endl;
    newFile << "6.5000E-11,0.0 " << std::endl;
    newFile << "*HYPERFOAM, N =  2" << std::endl;
    newFile << density_9 << std::endl;

    newFile << "*MATERIAL, NAME=pu_rg95_10kpa" << std::endl;
    newFile << "*DENSITY" << std::endl;
    newFile << "6.5000E-11,0.0 " << std::endl;
    newFile << "*HYPERFOAM, N =  2" << std::endl;
    newFile << density_10 << std::endl;

    newFile << "*****" << std::endl;
    newFile.close();
    infile.close();
}
void read_directory(const char *path, const char *folder_path, const std::vector<std::vector<int>> &permutations)
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
                for (int i = 0; i < permutations.size(); i++)
                {
                    std::string output_dirpath = folderpath + '/' + filepath.substr(0, filepath.size() - 4);
                    std::string output_filepath = output_dirpath + generate_filename(filepath, permutations[i]);
                    modify_inc(input_filepath, output_filepath, permutations[i]);
                }
            }
        }
        closedir(dir);
    }
    else
    {
        perror("Error in opening the directory");
    }
}
int main(int, char *argv[])
{
    std::string _start = argv[3];
    std::string _end = argv[4];
    std::string _step = argv[5];
    int start = stoi(_start);
    int end = stoi(_end);
    int step = stoi(_step);
    const char *out_dir = argv[1];
    const char *folder = argv[2];
    // std::vector<std::vector<int>> res = generate_permutations(start, end, step);
    std::vector<std::vector<int>> res{{8, 8, 10, 10}};
    read_directory(argv[1], argv[2], res);
    return 0;
}