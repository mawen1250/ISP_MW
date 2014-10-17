#ifndef MW_ARGS_H_
#define MW_ARGS_H_


#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>


class Args
{
private:
    static const int OK = 0;
    static const int Lack = 1;
    static const int Invalid = 2;

private:
    const int argc = 0;
    const std::vector<std::string> &args;

    int Flag = OK;

public:
    Args(const int _argc, const std::vector<std::string> &_args)
        : argc(_argc), args(_args) {}

    ~Args() {}

    void Check() const
    {
        if (Flag > 0) exit(EXIT_FAILURE);
    }

    template<typename T>
    void GetPara(int &i, T &para)
    {
        if (++i < argc)
        {
            para = std::stoi(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag |= Lack;
        }
    }

    template<>
    void GetPara(int &i, long &para)
    {
        if (++i < argc)
        {
            para = std::stol(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag |= Lack;
        }
    }

    template<>
    void GetPara(int &i, unsigned long &para)
    {
        if (++i < argc)
        {
            para = std::stoul(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag |= Lack;
        }
    }

    template<>
    void GetPara(int &i, long long &para)
    {
        if (++i < argc)
        {
            para = std::stoll(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag |= Lack;
        }
    }

    template<>
    void GetPara(int &i, unsigned long long &para)
    {
        if (++i < argc)
        {
            para = std::stoull(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag |= Lack;
        }
    }

    template<>
    void GetPara(int &i, float &para)
    {
        if (++i < argc)
        {
            para = std::stof(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag |= Lack;
        }
    }

    template<>
    void GetPara(int &i, double &para)
    {
        if (++i < argc)
        {
            para = std::stod(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag |= Lack;
        }
    }

    template<>
    void GetPara(int &i, long double &para)
    {
        if (++i < argc)
        {
            para = std::stold(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag |= Lack;
        }
    }

    template<>
    void GetPara(int &i, bool &para)
    {
        if (++i < argc)
        {
            std::string arg_temp(args[i]);
            std::transform(arg_temp.begin(), arg_temp.end(), arg_temp.begin(), tolower);
            if (arg_temp == "true")
            {
                para = true;
            }
            else if (arg_temp == "false")
            {
                para = false;
            }
            else
            {
                std::cout << "Invalid argument specified for option " << args[i - 1] << ", must be \"true\" or \"false\"!\n";

                Flag |= Invalid;
            }
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag |= Lack;
        }
    }

    template<>
    void GetPara(int &i, std::string &para)
    {
        if (++i < argc)
        {
            para = args[i];
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag |= Lack;
        }
    }
};


#endif
