#ifndef MW_ARGS_H_
#define MW_ARGS_H_


#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>


namespace mw {

    template<typename T>
    inline int arg2para(int & i, const int argc, const std::vector<std::string> &args, T & para)
    {
        int Flag = 0;

        if (++i < argc)
        {
            para = std::stoi(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int arg2para(int & i, const int argc, const std::vector<std::string> &args, long & para)
    {
        int Flag = 0;

        if (++i < argc)
        {
            para = std::stol(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int arg2para(int & i, const int argc, const std::vector<std::string> &args, unsigned long & para)
    {
        int Flag = 0;

        if (++i < argc)
        {
            para = std::stoul(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int arg2para(int & i, const int argc, const std::vector<std::string> &args, long long & para)
    {
        int Flag = 0;

        if (++i < argc)
        {
            para = std::stoll(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int arg2para(int & i, const int argc, const std::vector<std::string> &args, unsigned long long & para)
    {
        int Flag = 0;

        if (++i < argc)
        {
            para = std::stoull(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int arg2para(int & i, const int argc, const std::vector<std::string> &args, float & para)
    {
        int Flag = 0;

        if (++i < argc)
        {
            para = std::stof(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int arg2para(int & i, const int argc, const std::vector<std::string> &args, double & para)
    {
        int Flag = 0;

        if (++i < argc)
        {
            para = std::stod(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int arg2para(int & i, const int argc, const std::vector<std::string> &args, long double & para)
    {
        int Flag = 0;

        if (++i < argc)
        {
            para = std::stold(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int arg2para(int & i, const int argc, const std::vector<std::string> &args, bool & para)
    {
        int Flag = 0;

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

                Flag = 1;
            }
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int arg2para(int & i, const int argc, const std::vector<std::string> &args, std::string & para)
    {
        int Flag = 0;

        if (++i < argc)
        {
            para = args[i];
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

} // namespace mw


#endif