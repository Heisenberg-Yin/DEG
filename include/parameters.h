#ifndef STKQ_PARAMETERS_H
#define STKQ_PARAMETERS_H

#include <sstream>
#include <unordered_map>


// 这段代码定义了一个名为 Parameters 的类，用于存储和管理参数集
namespace stkq {
    class Parameters {
    public:
        template<typename T>
        inline void set(const std::string &name, const T &val) {
            std::stringstream ss;
            ss << val;
            params[name] = ss.str();
        }

        template<typename T>
        inline T get(const std::string &name) const {
            auto item = params.find(name);
            if (item == params.end()) {
                throw std::invalid_argument("Invalid paramter name : " + name + ".");
            } else {
                return ConvertStrToValue<T>(item->second);
            }
        }

        inline std::string toString() const {
            std::string res;
            for (auto &param : params) {
                res += param.first;
                res += ":";
                res += param.second;
                res += " ";
            }
            return res;
        }

    private:
        std::unordered_map<std::string, std::string> params;

        template<typename T>
        inline T ConvertStrToValue(const std::string &str) const {
            std::stringstream sstream(str);
            T value;
            if (!(sstream >> value) || !sstream.eof()) {
                std::stringstream err;
                err << "Fail to convert value" << str << " to type: " << typeid(value).name();
                throw std::runtime_error(err.str());
            }
            return value;
        }
    };
}

#endif
