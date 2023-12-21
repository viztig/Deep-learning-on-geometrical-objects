// Author:VAIBHAV VIKAS (VAVIKAS)
#pragma once
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <set>
#define space << " " <<
std::vector<std::vector<int>> generate_permutations(int start, int end, int step)
{
    std::set<std::vector<int>> v;
    for (int i = start; i <= end; i += step)
    {
        std::vector<int> a{6, 6, 7, 7};
        a[0] = i;
        for (int j = start; j <= end; j += step)
        {
            a[1] = j;
            for (int k = start; k <= end; k += step)
            {
                if (i < k && j < k)
                {
                    a[2] = k;
                }
                for (int l = start; l <= end; l += step)
                {
                    if (i < l && j < l)
                    {
                        a[3] = l;
                    }
                    if (a[0] < a[2] && a[1] < a[2] && a[0] < a[3] && a[1] < a[3])
                    {
                        v.insert(a);
                    }
                }
            }
        }
    }
    /*
    std::ofstream outFile(argv[1]);
    if (outFile.is_open())
    {
        std::streambuf *coutBuffer = std::cout.rdbuf();
        std::cout.rdbuf(outFile.rdbuf());
        for (auto a : v)
        {
            std::cout << a[0] space a[1] space a[2] space a[3] << std::endl;
        }
        std::cout.rdbuf(coutBuffer);
        outFile.close();
    }*/
    std::vector<std::vector<int>> res(v.begin(), v.end());
    return res;
}
/*int main()
{
    std::vector<std::vector<int>> res = generate_permutations(5, 10, 2);
    for (auto &it : res)
    {
        for (auto &x : it)
        {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }
    std::cout << res.size() << std::endl;
}*/