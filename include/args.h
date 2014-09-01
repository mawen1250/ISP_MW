#ifndef MW_ARGS_H_
#define MW_ARGS_H_


#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>


namespace mw {

    template<typename T>
    inline int args2arg(int & i, const int argc, std::string * args, T & arg)
    {
        int Flag = 0;

        if (++i<argc)
        {
            arg = std::stoi(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int args2arg(int & i, const int argc, std::string * args, long & arg)
    {
        int Flag = 0;

        if (++i<argc)
        {
            arg = std::stol(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int args2arg(int & i, const int argc, std::string * args, unsigned long & arg)
    {
        int Flag = 0;

        if (++i<argc)
        {
            arg = std::stoul(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int args2arg(int & i, const int argc, std::string * args, long long & arg)
    {
        int Flag = 0;

        if (++i<argc)
        {
            arg = std::stoll(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int args2arg(int & i, const int argc, std::string * args, unsigned long long & arg)
    {
        int Flag = 0;

        if (++i<argc)
        {
            arg = std::stoull(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int args2arg(int & i, const int argc, std::string * args, float & arg)
    {
        int Flag = 0;

        if (++i<argc)
        {
            arg = std::stof(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int args2arg(int & i, const int argc, std::string * args, double & arg)
    {
        int Flag = 0;

        if (++i<argc)
        {
            arg = std::stod(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int args2arg(int & i, const int argc, std::string * args, long double & arg)
    {
        int Flag = 0;

        if (++i<argc)
        {
            arg = std::stold(args[i]);
        }
        else
        {
            std::cout << "No argument specified for option " << args[i - 1] << "!\n";

            Flag = 1;
        }

        return Flag;
    }

    template<>
    inline int args2arg(int & i, const int argc, std::string * args, bool & arg)
    {
        int Flag = 0;

        if (++i<argc)
        {
            std::transform(args[i].begin(), args[i].end(), args[i].begin(), tolower);
            if (args[i] == "true")
            {
                arg = true;
            }
            else if (args[i] == "false")
            {
                arg = false;
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
    inline int args2arg(int & i, const int argc, std::string * args, std::string & arg)
    {
        int Flag = 0;

        if (++i<argc)
        {
            arg = args[i];
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