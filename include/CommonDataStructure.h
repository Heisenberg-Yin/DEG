#ifndef STKQ_COMMONDATASTRUCTURE_H
#define STKQ_COMMONDATASTRUCTURE_H

#include <iostream>
#include <string.h>
#include <memory>

namespace stkq {
    template<typename T>
    // 这段代码定义了一个模板类 Array，用于封装对数组的操作。它提供了一些功能，比如数组的创建、拷贝、赋值以及对数组元素的访问
    class Array
    {
        // Array 是一个模板类，可以用于任何类型 T 的数组。
    public:
        Array();

        Array(T* p_array, std::size_t p_length, bool p_transferOwnership);

        Array(T* p_array, std::size_t p_length, std::shared_ptr<T> p_dataHolder);

        Array(Array<T>&& p_right);

        Array(const Array<T>& p_right);

        Array<T>& operator= (Array<T>&& p_right);

        Array<T>& operator= (const Array<T>& p_right);

        T& operator[] (std::size_t p_index);

        const T& operator[] (std::size_t p_index) const;

        ~Array();

        T* Data() const;

        std::size_t Length() const;

        std::shared_ptr<T> DataHolder() const;

        void Set(T* p_array, std::size_t p_length, bool p_transferOwnership);

        void Clear();

        static Array<T> Alloc(std::size_t p_length);

        const static Array<T> c_empty;

    private:
        T* m_data; // m_data：指向数组数据的指针

        std::size_t m_length; //m_length：数组的长度

        // Notice this is holding an array. Set correct deleter for this.
        std::shared_ptr<T> m_dataHolder; 
        //一个 std::shared_ptr<T> 智能指针 用于管理数组的内存 注意这里它被用来持有一个数组 所以在重置时需要提供正确的删除器 
        /*
        std::shared_ptr<T> 是 C++ 标准库中提供的一种智能指针，它用于自动管理一个对象的生命周期，确保对象在不再被需要时被适时地销毁
        std::shared_ptr<T> 的主要特点和作用包括：        
        1) 自动内存管理：
        std::shared_ptr 通过引用计数机制管理其指向的对象。当新的 shared_ptr 指向同一个对象时，引用计数增加；
        当 shared_ptr 被销毁或者重新指向另一个对象时，引用计数减少。当引用计数降至零时，它指向的对象会被自动销毁（调用析构函数），并释放相关资源
        2) 共享所有权：
        std::shared_ptr 允许多个 shared_ptr 实例共享对同一个对象的所有权。这意味着多个 shared_ptr 可以指向同一个对象，
        而对象只会在最后一个引用它的 shared_ptr 被销毁时被删除
        3) 避免内存泄漏：
        由于 shared_ptr 自动管理内存，它有助于避免内存泄漏，这在使用原始指针时是一个常见问题。
        4) 自定义删除器：
        shared_ptr 允许用户指定自定义删除器。这在管理非内存资源（如文件句柄、网络连接等）时特别有用
        */
    };

    template<typename T>
    const Array<T> Array<T>::c_empty;


    template<typename T>
    Array<T>::Array()
            : m_data(nullptr),
              m_length(0)
    {
    }

    template<typename T>
    Array<T>::Array(T* p_array, std::size_t p_length, bool p_transferOnwership)

            : m_data(p_array),
              m_length(p_length)
    {
        // bool p_transferOwnership：一个布尔值，指示 Array 对象是否应该接管 p_array 指向的数组的所有权
        if (p_transferOnwership)
        {
            // 如果 p_transferOwnership 为 true，则创建一个 std::shared_ptr （m_dataHolder）来管理 p_array 指向的数组
            m_dataHolder.reset(m_data, std::default_delete<T[]>());
            // 使用 std::default_delete<T[]>() 作为删除器，这意味着当 shared_ptr 被销毁或重置时，它将使用 delete[] 来释放数组
        }
    }


    template<typename T>
    Array<T>::Array(T* p_array, std::size_t p_length, std::shared_ptr<T> p_dataHolder)
            : m_data(p_array),
              m_length(p_length),
              m_dataHolder(std::move(p_dataHolder))
    {
        //m_dataHolder 通过 std::move(p_dataHolder) 接管传入的 shared_ptr
        //这里使用 std::move 是为了避免复制 shared_ptr（这会增加引用计数），而是直接接管其所有权
    }



    // 移动构造函数
    // 输入类型：Array<T>&& 表示右值引用。右值引用通常绑定到临时对象或那些即将被销毁的对象，这意味着它们不会再被其他部分代码使用。
    // 含义：右值引用的使用标志着移动构造函数可以安全地“移动”资源，而不是复制，因为原始对象（即函数参数 p_right）之后不会再被使用
    template<typename T>
    Array<T>::Array(Array<T>&& p_right)
            : m_data(p_right.m_data),
              m_length(p_right.m_length),
              m_dataHolder(std::move(p_right.m_dataHolder))
    {
    }

    //拷贝构造函数
    // 输入类型：const Array<T>& 表示常量左值引用。左值引用通常指向长期存储的值，可能会在之后的代码中继续使用。
    // 含义：常量左值引用的使用表明拷贝构造函数需要保留原始对象（即函数参数 p_right）的状态不变，并创建一个新的共享同一资源的对象    
    template<typename T>
    Array<T>::Array(const Array<T>& p_right)
            : m_data(p_right.m_data),
              m_length(p_right.m_length),
              m_dataHolder(p_right.m_dataHolder)
    {
    }


    template<typename T>
    Array<T>&
    Array<T>::operator= (Array<T>&& p_right)
    {
        m_data = p_right.m_data;
        m_length = p_right.m_length;
        m_dataHolder = std::move(p_right.m_dataHolder);

        return *this;
    }


    template<typename T>
    Array<T>&
    Array<T>::operator= (const Array<T>& p_right)
    {
        m_data = p_right.m_data;
        m_length = p_right.m_length;
        m_dataHolder = p_right.m_dataHolder;

        return *this;
    }


    template<typename T>
    T&
    Array<T>::operator[] (std::size_t p_index)
    {
        return m_data[p_index];
    }


    template<typename T>
    const T&
    Array<T>::operator[] (std::size_t p_index) const
    {
        return m_data[p_index];
    }


    template<typename T>
    Array<T>::~Array()
    {
    }


    template<typename T>
    T*
    Array<T>::Data() const
    {
        return m_data;
    }


    template<typename T>
    std::size_t
    Array<T>::Length() const
    {
        return m_length;
    }


    template<typename T>
    std::shared_ptr<T>
    Array<T>::DataHolder() const
    {
        return m_dataHolder;
    }


    template<typename T>
    void
    Array<T>::Set(T* p_array, std::size_t p_length, bool p_transferOwnership)
    {
        m_data = p_array;
        m_length = p_length;

        if (p_transferOwnership)
        {
            m_dataHolder.reset(m_data, std::default_delete<T[]>());
        }
    }


    template<typename T>
    void
    Array<T>::Clear()
    {
        m_data = nullptr;
        m_length = 0;
        m_dataHolder.reset();
    }


    template<typename T>
    Array<T>
    Array<T>::Alloc(std::size_t p_length)
    {
        Array<T> arr;
        if (0 == p_length)
        {
            return arr;
        }

        arr.m_dataHolder.reset(new T[p_length], std::default_delete<T[]>());

        arr.m_length = p_length;
        arr.m_data = arr.m_dataHolder.get();
        return arr;
    }


    typedef Array<std::uint8_t> ByteArray;
}

#endif 