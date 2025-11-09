#include <ATen/core/functional.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <xsigma/util/hash.h>

namespace torch::jit
{

namespace
{
size_t hashType(const Type& type)
{
    if (auto named_type = type.castRaw<ClassType>())
    {
        return xsigma::get_hash(named_type->name().value(), named_type->compilation_unit());
    }
    size_t hash = 0;
    for (const auto& containedType : type.containedTypes())
    {
        hash = xsigma::hash_combine(hash, hashType(*containedType));
    }
    hash = xsigma::hash_combine(hash, get_hash(type.kind()));
    return hash;
}
}  // namespace

size_t HashType::operator()(const TypePtr& type) const
{
    return hashType(*type);
}

size_t HashType::operator()(const xsigma::ConstTypePtr& type) const
{
    return hashType(*type);
}

bool EqualType::operator()(const TypePtr& a, const TypePtr& b) const
{
    return *a == *b;
}

bool EqualType::operator()(const xsigma::ConstTypePtr& a, const xsigma::ConstTypePtr& b) const
{
    return *a == *b;
}

}  // namespace torch::jit
