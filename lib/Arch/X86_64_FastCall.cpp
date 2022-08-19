#include <anvill/Declarations.h>
#include<glog/logging.h>
#include <remill/Arch/Arch.h>
#include <remill/Arch/Name.h>

#include "AllocationState.h"
#include "Arch.h"

namespace anvill {
  namespace {

    static const std::vector< RegisterConstraint > kParamRegConstraints = {

      // First integer argument
      //
      RegisterConstraint( {
        VariantConstraint( "CL", kTypeIntegral, kMaxBit8 ),
        VariantConstraint( "CX", kTypeIntegral, kMaxBit16 ),
        VariantConstraint( "ECX", kTypeIntegral, kMaxBit32 ),
        VariantConstraint( "RCX", kTypeIntegral, kMaxBit64 )
      } ),
      // Second integer argument
      //
      RegisterConstraint( {
        VariantConstraint( "DL", kTypeIntegral, kMaxBit8 ),
        VariantConstraint( "DX", kTypeIntegral, kMaxBit16 ),
        VariantConstraint( "EDX", kTypeIntegral, kMaxBit32 ),
        VariantConstraint( "RDX", kTypeIntegral, kMaxBit64 )
      } ),
      // Third integer argument
      //
      RegisterConstraint( {
        VariantConstraint( "R8B", kTypeIntegral, kMaxBit8 ),
        VariantConstraint( "R8W", kTypeIntegral, kMaxBit16 ),
        VariantConstraint( "R8D", kTypeIntegral, kMaxBit32 ),
        VariantConstraint( "R8", kTypeIntegral, kMaxBit64 )
      } ),
      // Fourth integer argument
      //
      RegisterConstraint( {
        VariantConstraint( "R9B", kTypeIntegral, kMaxBit8 ),
        VariantConstraint( "R9W", kTypeIntegral, kMaxBit16 ),
        VariantConstraint( "R9D", kTypeIntegral, kMaxBit32 ),
        VariantConstraint( "R9", kTypeIntegral, kMaxBit64 )
      } ),

      // First floating-point argument
      //
      RegisterConstraint( {
        VariantConstraint( "XMM0", kTypeFloat, kMaxBit128 )
      } ),
      // First vector-type argument
      RegisterConstraint( {
        VariantConstraint( "YMM0", kTypeVec, kMaxBit128 )
      } ),
      // Second floating-point argument
      //
      RegisterConstraint( {
        VariantConstraint( "XMM1", kTypeFloat, kMaxBit128 )
      } ),
      // Second vector-type argument
      //
      RegisterConstraint( {
        VariantConstraint( "YMM1", kTypeVec, kMaxBit128 )
      } ),
      // Third floating-point argument
      //
      RegisterConstraint( {
        VariantConstraint( "XMM2", kTypeFloat, kMaxBit128 )
      } ),
      // Third vector-type argument
      //
      RegisterConstraint( {
        VariantConstraint( "YMM2", kTypeVec, kMaxBit128 )
      } ),
      // Fourth floating-point argument
      //
      RegisterConstraint( {
        VariantConstraint( "XMM3", kTypeFloat, kMaxBit128 )
      } ),
      // Fourth vector-type argument
      //
      RegisterConstraint( {
        VariantConstraint( "YMM3", kTypeVec, kMaxBit128 )
      } )

    };

    static const std::vector< RegisterConstraint > kAVXParamRegConstraints =
      ApplyX86Ext( kParamRegConstraints, remill::kArchAMD64_AVX );
    static const std::vector< RegisterConstraint > kAVX512ParamRegConstraints =
      ApplyX86Ext( kParamRegConstraints, remill::kArchAMD64_AVX512 );

    static const std::vector< RegisterConstraint > kReturnRegConstraints = {
      // Integer type return register
      //
      RegisterConstraint( {
        VariantConstraint( "AL", kTypeIntegral, kMaxBit8 ),
        VariantConstraint( "AX", kTypeIntegral, kMaxBit16 ),
        VariantConstraint( "EAX", kTypeIntegral, kMaxBit32 ),
        VariantConstraint( "RAX", kTypeIntegral, kMaxBit64 )
      } ),

      // Floating-point return return register
      //
      RegisterConstraint( {
        VariantConstraint( "XMM0", kTypeFloatOrVec, kMaxBit128 )
      } )
    };

    static const std::vector< RegisterConstraint > kAVXReturnRegConstraints =
      ApplyX86Ext( kReturnRegConstraints, remill::kArchAMD64_AVX );
    static const std::vector< RegisterConstraint > kAVX512ReturnRegConstraints =
      ApplyX86Ext( kReturnRegConstraints, remill::kArchAMD64_AVX512 );

  }

  class X86_64_FastCall : public CallingConvention {
    public:
      explicit X86_64_FastCall( const remill::Arch *arch );
      virtual ~X86_64_FastCall( void ) = default;

      llvm::Error AllocateSignature( FunctionDecl &fdecl, llvm::Function &func ) override;

    private:
      llvm::Error BindParameters( llvm::Function &function, bool injected_sret, std::vector< ParameterDecl > &param_decls );

      llvm::Error BindReturnValues( llvm::Function &function, bool &injected_sret, std::vector< ValueDecl > &ret_decls );

      const std::vector< RegisterConstraint > &parameter_register_constraints;
      const std::vector< RegisterConstraint > &return_register_constraints;
  };

  std::unique_ptr< CallingConvention > CallingConvention::CreateX86_64_FastCall( const remill::Arch *arch ) {
    return std::make_unique< CallingConvention >( new X86_64_FastCall( arch ) );
  }

  X86_64_FastCall::X86_64_FastCall( const remill::Arch *arch )
    : CallingConvention( llvm::CallingConv::Win64, arch ),
      parameter_register_constraints(
        SelectX86Constraint( arch->arch_name, kParamRegConstraints, kAVXParamRegConstraints, kAVX512ParamRegConstraints )
      ),
      return_register_constraints(
        SelectX86Constraint( arch->arch_name, kReturnRegConstraints, kAVXReturnRegConstraints, kAVX512ReturnRegConstraints )
      ) {}

  // Allocates the elements of the function signature of func to memory or registers
  // This includs parameters / arguments, return values and the return stack pointer.
  //
  llvm::Error X86_64_FastCall::AllocateSignature( FunctionDecl &fdecl, llvm::Function &func ) {
    bool injected_sret = false;

    auto err = BindReturnValues( func, injected_sret, fdecl.returns );
    if ( remill::IsError( err ) ) {
      return err;
    }

    err = BindParameters( func, injected_sret, fdecl.params );
    if ( remill::IsError( err ) ) {
      return err;
    }

    fdecl.return_stack_pointer_offset = 8;
    fdecl.return_stack_pointer = arch->RegisterByName( "RSP" );

    fdecl.return_address.mem_reg = fdecl.return_stack_pointer;
    fdecl.return_address.mem_offset = 0;
    fdecl.return_address.type = fdecl.return_stack_pointer->type;

    return llvm::Error::success();
  }

  llvm::Error X86_64_FastCall::BindReturnValues( llvm::Function &function, bool &injected_sret, std::vector< ValueDecl > &ret_values ) {
    llvm::Type *ret_type = function.getReturnType();
    injected_sret = false;

    // If there is a Struct ret parameter it has to be handled specifically.
    // The return pointer is guaranteed to be in RAX, depending on the return value size
    // (the struct size) it is either returned by value (size <= 64 bits, RAX will hold the value) or promoted
    // to a pointer (size > 64 bits, RAX will hold the pointer)
    //
    if ( function.hasStructRetAttr() ) {
      auto &value_declaration = ret_values.emplace_back();

      llvm::DataLayout dl( function.getParent() );
      const auto bit_width = dl.getTypeSizeInBits( ret_type );
      if ( bit_width <= 64 ) {
        value_declaration.type = ret_type;
      } else {
        value_declaration.type = llvm::PointerType::get( function.getContext(), 0 );
      }

      value_declaration.reg = arch->RegisterByName( "RAX" );
      return llvm::Error::success();
    }

    switch ( ret_type->getTypeID() )
    {
      case llvm::Type::VoidTyID: {
        return llvm::Error::success();
      }

      case llvm::Type::IntegerTyID: {
        const auto *int_ty = llvm::dyn_cast< llvm::IntegerType >( ret_type );
        const auto bit_width = int_ty->getBitWidth();

        // Scalar value can fit into 64 bit, put it into RAX
        //
        if ( bit_width <= 64 ) {
          auto &value_declaration = ret_values.emplace_back();
          value_declaration.reg = arch->RegisterByName( "RAX" );
          value_declaration.type = ret_type;
          return llvm::Error::success();
        }
        // Else if less than 128 bit return it through XMM0
        //
        else if ( bit_width <= 128 ) {
          auto &value_declaration = ret_values.emplace_back();
          value_declaration.reg = arch->RegisterByName( "XMM0" );
          value_declaration.type = ret_type;
          return llvm::Error::success();
        }
        // Otherwise inject struct ret, pointer is returned into RAX
        //
        else {
          injected_sret = true;

          auto &value_declaration = ret_values.emplace_back();
          value_declaration.reg = arch->RegisterByName( "RAX" );
          value_declaration.type = llvm::PointerType::get( function.getContext(), 0 );
          return llvm::Error::success();
        }
      }

      // Pointers always fit into RAX
      //
      case llvm::Type::PointerTyID: {
        auto &value_declaration = ret_values.emplace_back();
        value_declaration.reg = arch->RegisterByName( "RAX" );
        value_declaration.type = ret_type;
        return llvm::Error::success();
      }

      // Floats and doubles go in XMM0
      //
      case llvm::Type::FloatTyID:
      case llvm::Type::DoubleTyID: {
        auto &value_declaration = ret_values.emplace_back();
          value_declaration.reg = arch->RegisterByName( "XMM0" );
          value_declaration.type = ret_type;
          return llvm::Error::success();
      }

      case llvm::Type::X86_MMXTyID: {
        auto &value_declaration = ret_values.emplace_back();
        value_declaration.reg = arch->RegisterByName( "MM0" );
        value_declaration.type = ret_type;
        return llvm::Error::success();
      }

      case llvm::Type::X86_FP80TyID: {
        auto &value_declaration = ret_values.emplace_back();
        value_declaration.reg = arch->RegisterByName( "ST0" );
        value_declaration.type = ret_type;
        return llvm::Error::success();
      }

      case llvm::Type::FixedVectorTyID:
      case llvm::Type::ArrayTyID:
      case llvm::Type::StructTyID: {
        injected_sret = true;

        auto &value_declaration = ret_values.emplace_back();
        value_declaration.reg = arch->RegisterByName( "RAX" );
        value_declaration.type = llvm::PointerType::get( function.getContext(), 0 );
        return llvm::Error::success();
      }

      default:
        return llvm::createStringError(
          std::errc::invalid_argument,
          "Could not allocate unsupported type '%s' to return register in function '%s'",
          remill::LLVMThingToString( ret_type ).c_str(),
          function.getName().str().c_str()
        );
    }
  }

  llvm::Error X86_64_FastCall::BindParameters( llvm::Function &function, bool injected_sret, std::vector< ParameterDecl > &parameter_declarations ) {
    const auto param_names = TryRecoverParamNames( function );
    llvm::DataLayout dl( function.getParent() );

    // Track allocated registers
    //
    AllocationState alloc_param( parameter_register_constraints, arch, this );

    // Stack offset describes the stack position of the first stack argument on
    // entry to the callee. For X86_64_FastCall, this is [rsp + 8] since there is the
    // return address at [rsp].
    //
    uint64_t stack_offset = 8;

    // If there is an injected sret (an implicit sret) then we need to allocate
    // the first parameter to the sret struct. The type of said sret parameter
    // will be the return type of the function.
    //
    if ( injected_sret ) {
      auto &decl = parameter_declarations.emplace_back();

      decl.name = "__struct_ret_ptr";
      decl.type = function.getReturnType();
      decl.reg = arch->RegisterByName( "RAX" );
      alloc_param.reserved[ 0 ] = true;
    }

    const auto rsp_reg = arch->RegisterByName( "RSP" );

    for ( auto &argument : function.args() ) {
      const auto &param_name = param_names[ argument.getArgNo() ];
      const auto param_type = argument.getType();

      // Try to allocate from a register
      //
      if ( auto allocation = alloc_param.TryBasicRegisterAllocate( *param_type, llvm::None ) ) {
        auto prev_size = parameter_declarations.size();

        for ( const auto &param_decl : allocation.getValue() ) {
          auto &declaration = parameter_declarations.emplace_back();
          declaration.type = param_decl.type;

          if ( param_decl.reg ) {
            declaration.reg = param_decl.reg;
          } else {
            declaration.mem_offset = param_decl.mem_offset;
            declaration.mem_reg = param_decl.mem_reg;
          }
        }

        if ( !param_name.empty() ) {
          parameter_declarations[ prev_size ].name = param_name;
        }
      }
      // Else allocate from the stack
      //
      else {
        auto &declaration = parameter_declarations.emplace_back();
        declaration.type = param_type;
        declaration.mem_offset = static_cast< int64_t >( stack_offset );
        declaration.mem_reg = rsp_reg;
        stack_offset += dl.getTypeAllocSize( argument.getType() );

        if ( !param_name.empty() ) {
          declaration.name = param_name;
        }
      }
    }

    return llvm::Error::success();
  }
}
