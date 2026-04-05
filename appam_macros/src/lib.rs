//! Procedural macros for the Appam AI agent framework.
//!
//! This crate provides the `#[tool]` attribute macro for easily defining tools
//! that can be used by AI agents with support for parameter descriptions and defaults.
//!
//! # Examples
//!
//! ```ignore
//! use appam_macros::tool;
//!
//! #[tool(description = "Echoes back the input message")]
//! fn echo(
//!     #[arg(description = "Message to echo")]
//!     message: String,
//! ) -> anyhow::Result<String> {
//!     Ok(message)
//! }
//!
//! // With optional parameters
//! #[tool(description = "Search with options")]
//! fn search(
//!     #[arg(description = "Search query")]
//!     query: String,
//!
//!     #[arg(description = "Max results", default = 10)]
//!     max_results: u32,
//! ) -> anyhow::Result<Vec<String>> {
//!     // Implementation
//!     Ok(vec![])
//! }
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::punctuated::Punctuated;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Attribute, Data, DeriveInput, Fields, FnArg, ItemFn, Lit, Meta, Pat,
    PatType, Token, Type, Variant,
};

/// Custom attribute parser for syn 2.0
struct ToolAttributes {
    attrs: Punctuated<Meta, Token![,]>,
}

impl Parse for ToolAttributes {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(ToolAttributes {
            attrs: input.parse_terminated(Meta::parse, Token![,])?,
        })
    }
}

/// Attribute macro for defining AI agent tools.
///
/// This macro generates a complete `Tool` trait implementation from a function,
/// automatically creating JSON schemas and handling argument marshaling.
///
/// # Attributes
///
/// - `name`: The tool name (defaults to function name)
/// - `description`: Human-readable description for the LLM
///
/// # Supported Function Signatures
///
/// The macro supports various function signatures:
///
/// ```ignore
/// // Single string parameter
/// #[tool(description = "Echo tool")]
/// fn echo(message: String) -> Result<String> { ... }
///
/// // Multiple parameters
/// #[tool(description = "Calculator")]
/// fn calculate(a: f64, b: f64, operation: String) -> Result<f64> { ... }
///
/// // No return value
/// #[tool(description = "Logger")]
/// fn log_message(message: String) -> Result<()> { ... }
///
/// // JSON value parameters and return
/// #[tool(description = "Complex tool")]
/// fn complex(args: serde_json::Value) -> Result<serde_json::Value> { ... }
/// ```
///
/// # Generated Code
///
/// The macro generates:
/// - A struct with the same name as the function (converted to PascalCase)
/// - A `Tool` trait implementation
/// - A constructor function with the original function name
/// - JSON schema generation based on parameter types
///
/// # Type Mapping
///
/// Rust types are mapped to JSON schema types:
/// - `String` → `"string"`
/// - `i32`, `i64`, `u32`, `u64`, `f32`, `f64` → `"number"`
/// - `bool` → `"boolean"`
/// - `Vec<T>` → `"array"`
/// - `serde_json::Value` → pass-through
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_args = parse_macro_input!(attr as ToolAttributes);
    let input_fn = parse_macro_input!(item as ItemFn);

    match tool_impl(attr_args.attrs, input_fn) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error().into(),
    }
}

/// Parameter information with attributes
struct ParamInfo {
    name: String,
    ty: Box<Type>,
    ident: syn::Ident,
    description: Option<String>,
    default: Option<syn::Expr>,
    json_type: &'static str,
}

fn tool_impl(attr_args: Punctuated<Meta, Token![,]>, input_fn: ItemFn) -> syn::Result<TokenStream> {
    // Parse tool-level attributes
    let mut tool_name: Option<String> = None;
    let mut description: Option<String> = None;

    for meta in attr_args {
        match meta {
            Meta::NameValue(nv) => {
                let ident = nv
                    .path
                    .get_ident()
                    .ok_or_else(|| syn::Error::new_spanned(&nv.path, "Expected identifier"))?;

                match ident.to_string().as_str() {
                    "name" => {
                        if let syn::Expr::Lit(expr_lit) = &nv.value {
                            if let Lit::Str(lit) = &expr_lit.lit {
                                tool_name = Some(lit.value());
                            }
                        }
                    }
                    "description" => {
                        if let syn::Expr::Lit(expr_lit) = &nv.value {
                            if let Lit::Str(lit) = &expr_lit.lit {
                                description = Some(lit.value());
                            }
                        }
                    }
                    _ => {
                        return Err(syn::Error::new_spanned(
                            ident,
                            "Unknown attribute. Supported: name, description",
                        ));
                    }
                }
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    meta,
                    "Expected name=value attribute",
                ));
            }
        }
    }

    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.vis;
    let fn_block = &input_fn.block;

    // Use provided name or convert function name
    let tool_name_str = tool_name.unwrap_or_else(|| fn_name.to_string());
    let description_str = description.unwrap_or_else(|| format!("Tool: {}", tool_name_str));

    // Generate struct name (PascalCase)
    let struct_name = syn::Ident::new(&to_pascal_case(&tool_name_str), fn_name.span());

    // Parse function parameters with their #[arg(...)] attributes
    let mut params: Vec<ParamInfo> = Vec::new();

    for arg in &input_fn.sig.inputs {
        if let FnArg::Typed(PatType { attrs, pat, ty, .. }) = arg {
            if let Pat::Ident(pat_ident) = &**pat {
                let param_name = pat_ident.ident.to_string();
                let json_type = type_to_json_type(ty);

                // Parse #[arg(...)] attributes
                let (arg_description, arg_default) = parse_arg_attributes(attrs)?;

                params.push(ParamInfo {
                    name: param_name,
                    ty: ty.clone(),
                    ident: pat_ident.ident.clone(),
                    description: arg_description,
                    default: arg_default,
                    json_type,
                });
            }
        }
    }

    // Check if function takes a single serde_json::Value parameter
    let takes_json_value =
        params.len() == 1 && params[0].json_type == "object" && is_json_value_type(&params[0].ty);

    // Check if function takes a single typed struct (for hybrid approach)
    // If it's not String/bool/number/Value, assume it's a custom struct with JsonSchema
    let takes_typed_struct = params.len() == 1
        && params[0].json_type == "string"  // Unknown types map to "string" by default
        && !is_primitive_type(&params[0].ty)
        && !is_json_value_type(&params[0].ty);

    // Generate parameter schema
    let param_schema = if takes_typed_struct {
        // For typed structs, generate schema at runtime using schemars
        let ty = &params[0].ty;
        quote! {
            {
                use ::schemars::JsonSchema;
                let root_schema = ::schemars::schema_for!(#ty);
                ::serde_json::to_value(&root_schema.schema)
                    .unwrap_or_else(|_| {
                        let mut map = ::serde_json::Map::new();
                        map.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                        map.insert("properties".to_string(), ::serde_json::Value::Object(::serde_json::Map::new()));
                        ::serde_json::Value::Object(map)
                    })
            }
        }
    } else {
        generate_param_schema(&params)
    };

    // Generate argument parsing code
    let arg_parsing = if takes_json_value {
        // Direct pass-through for serde_json::Value
        let param_ident = &params[0].ident;
        quote! {
            let #param_ident = args;
        }
    } else if takes_typed_struct {
        // Deserialize the entire args object into the typed struct
        let param_ident = &params[0].ident;
        let ty = &params[0].ty;
        quote! {
            let #param_ident: #ty = ::serde_json::from_value(args)
                .map_err(|e| ::anyhow::anyhow!("Failed to deserialize arguments: {}", e))?;
        }
    } else if params.is_empty() {
        // No parameters
        quote! {}
    } else {
        // Parse individual parameters based on their types
        let mut extractions = Vec::new();
        for param in &params {
            let name_ident = &param.ident;
            let name_str = &param.name;
            let ty = &param.ty;
            let has_default = param.default.is_some();

            let extraction = match param.json_type {
                "string" => {
                    if has_default {
                        let default_val = param.default.as_ref().unwrap();
                        quote! {
                            let #name_ident: #ty = args.get(#name_str)
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| #default_val.to_string());
                        }
                    } else {
                        quote! {
                            let #name_ident: #ty = args.get(#name_str)
                                .and_then(|v| v.as_str())
                                .ok_or_else(|| anyhow::anyhow!("Missing or invalid string parameter: {}", #name_str))?
                                .to_string();
                        }
                    }
                }
                "number" => {
                    if has_default {
                        let default_val = param.default.as_ref().unwrap();
                        quote! {
                            let #name_ident: #ty = args.get(#name_str)
                                .and_then(|v| v.as_f64())
                                .map(|n| n as #ty)
                                .unwrap_or(#default_val);
                        }
                    } else {
                        quote! {
                            let #name_ident: #ty = args.get(#name_str)
                                .and_then(|v| v.as_f64())
                                .ok_or_else(|| anyhow::anyhow!("Missing or invalid number parameter: {}", #name_str))? as #ty;
                        }
                    }
                }
                "boolean" => {
                    if has_default {
                        let default_val = param.default.as_ref().unwrap();
                        quote! {
                            let #name_ident: #ty = args.get(#name_str)
                                .and_then(|v| v.as_bool())
                                .unwrap_or(#default_val);
                        }
                    } else {
                        quote! {
                            let #name_ident: #ty = args.get(#name_str)
                                .and_then(|v| v.as_bool())
                                .ok_or_else(|| anyhow::anyhow!("Missing or invalid boolean parameter: {}", #name_str))?;
                        }
                    }
                }
                _ => {
                    // For complex types (structs), try to deserialize from JSON
                    if has_default {
                        let default_val = param.default.as_ref().unwrap();
                        quote! {
                            let #name_ident: #ty = args.get(#name_str)
                                .and_then(|v| serde_json::from_value(v.clone()).ok())
                                .unwrap_or(#default_val);
                        }
                    } else {
                        quote! {
                            let #name_ident: #ty = args.get(#name_str)
                                .ok_or_else(|| anyhow::anyhow!("Missing parameter: {}", #name_str))
                                .and_then(|v| serde_json::from_value(v.clone())
                                    .map_err(|e| anyhow::anyhow!("Failed to parse parameter {}: {}", #name_str, e)))?;
                        }
                    }
                }
            };

            extractions.push(extraction);
        }
        quote! {
            #(#extractions)*
        }
    };

    // Collect parameter identifiers for function call (unused but may be needed for future enhancements)
    let _param_idents: Vec<_> = params.iter().map(|p| &p.ident).collect();

    // Generate result handling code - always wrap in Result since tools return Result
    let result_handling = quote! {
        let result = { #fn_block };
        match result {
            Ok(value) => Ok(serde_json::json!({ "output": value })),
            Err(e) => Err(e),
        }
    };

    // Generate the tool struct and implementation
    let expanded = quote! {
        #[doc = #description_str]
        #fn_vis struct #struct_name;

        impl #struct_name {
            /// Create a new instance of this tool.
            #fn_vis fn new() -> Self {
                Self
            }
        }

        impl ::appam::tools::Tool for #struct_name {
            fn name(&self) -> &str {
                #tool_name_str
            }

            fn spec(&self) -> ::anyhow::Result<::appam::llm::ToolSpec> {
                let parameters = #param_schema;
                Ok(::appam::llm::ToolSpec {
                    type_field: "function".to_string(),
                    name: #tool_name_str.to_string(),
                    description: #description_str.to_string(),
                    parameters,
                    strict: None,
                })
            }

            fn execute(&self, args: ::serde_json::Value) -> ::anyhow::Result<::serde_json::Value> {
                #arg_parsing
                #result_handling
            }
        }

        /// Create a new instance of the tool.
        #fn_vis fn #fn_name() -> #struct_name {
            #struct_name::new()
        }
    };

    Ok(expanded.into())
}

/// Convert a function name to PascalCase for the struct name.
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}

/// Map Rust types to JSON schema types.
fn type_to_json_type(ty: &Type) -> &'static str {
    // This is a simplified version - in a real implementation,
    // we'd need more sophisticated type analysis
    let type_str = quote!(#ty).to_string();

    if type_str.contains("String") || type_str.contains("&str") {
        "string"
    } else if type_str.contains("bool") {
        "boolean"
    } else if type_str.contains("i32")
        || type_str.contains("i64")
        || type_str.contains("u32")
        || type_str.contains("u64")
        || type_str.contains("f32")
        || type_str.contains("f64")
    {
        "number"
    } else if type_str.contains("Vec") {
        "array"
    } else if type_str.contains("Value") {
        "object"
    } else {
        "string" // Default fallback
    }
}

/// Check if a type is serde_json::Value.
fn is_json_value_type(ty: &Type) -> bool {
    let type_str = quote!(#ty).to_string();
    type_str.contains("Value") || type_str.contains("serde_json :: Value")
}

/// Check if a type is a primitive Rust type.
fn is_primitive_type(ty: &Type) -> bool {
    let type_str = quote!(#ty).to_string();

    // Primitive types that we handle specially
    type_str == "String"
        || type_str.contains("&str")
        || type_str == "bool"
        || type_str == "i32"
        || type_str == "i64"
        || type_str == "u32"
        || type_str == "u64"
        || type_str == "f32"
        || type_str == "f64"
        || type_str == "usize"
        || type_str == "isize"
}

/// Parse #[arg(...)] attributes on function parameters
fn parse_arg_attributes(attrs: &[Attribute]) -> syn::Result<(Option<String>, Option<syn::Expr>)> {
    let mut description = None;
    let mut default = None;

    for attr in attrs {
        if !attr.path().is_ident("arg") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("description") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                description = Some(s.value());
                Ok(())
            } else if meta.path.is_ident("default") {
                let value = meta.value()?;
                let expr: syn::Expr = value.parse()?;
                default = Some(expr);
                Ok(())
            } else {
                Err(meta.error("Unknown arg attribute. Supported: description, default"))
            }
        })?;
    }

    Ok((description, default))
}

/// Generate JSON schema for parameters.
fn generate_param_schema(params: &[ParamInfo]) -> proc_macro2::TokenStream {
    if params.is_empty() {
        return quote! {
            ::serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            })
        };
    }

    let properties: Vec<_> = params
        .iter()
        .map(|param| {
            let name = &param.name;
            let json_type = param.json_type;
            let description = param.description.as_deref().unwrap_or("Parameter");

            quote! {
                #name: {
                    "type": #json_type,
                    "description": #description
                }
            }
        })
        .collect();

    // Only required parameters (those without defaults) go in "required" array
    let required: Vec<_> = params
        .iter()
        .filter(|p| p.default.is_none())
        .map(|p| {
            let name = &p.name;
            quote!(#name)
        })
        .collect();

    quote! {
        ::serde_json::json!({
            "type": "object",
            "properties": {
                #(#properties),*
            },
            "required": [#(#required),*]
        })
    }
}

/// Derive macro for generating JSON schemas with simplified description attributes.
///
/// This macro provides a cleaner API by hiding schemars as an internal implementation detail.
/// Users can use `#[description = "..."]` instead of `#[schemars(description = "...")]`.
///
/// # Examples
///
/// ```ignore
/// use appam::prelude::*;
///
/// #[derive(Deserialize, Schema)]
/// struct FileInput {
///     #[description = "Path to the file"]
///     file_path: String,
///
///     #[description = "Whether to create parent directories"]
///     create_dirs: bool,
/// }
///
/// #[derive(Deserialize, Schema)]
/// #[serde(rename_all = "lowercase")]
/// enum SortOrder {
///     #[description = "Sort by relevance"]
///     Relevance,
///     #[description = "Sort by date"]
///     Date,
/// }
/// ```
#[proc_macro_derive(Schema, attributes(description))]
pub fn derive_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match schema_impl(input) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error().into(),
    }
}

/// Implementation of the Schema derive macro
fn schema_impl(input: DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    match &input.data {
        Data::Struct(data_struct) => {
            let field_schemas = generate_field_schemas(&data_struct.fields)?;

            Ok(quote! {
                impl #impl_generics ::schemars::JsonSchema for #name #ty_generics #where_clause {
                    fn schema_name() -> String {
                        stringify!(#name).to_string()
                    }

                    fn json_schema(gen: &mut ::schemars::gen::SchemaGenerator) -> ::schemars::schema::Schema {
                        use ::schemars::schema::*;

                        let mut schema_object = SchemaObject {
                            instance_type: Some(InstanceType::Object.into()),
                            ..Default::default()
                        };

                        let mut properties = ::schemars::Map::new();
                        let mut required = ::std::collections::BTreeSet::new();

                        #field_schemas

                        schema_object.object = Some(Box::new(ObjectValidation {
                            properties,
                            required,
                            ..Default::default()
                        }));

                        Schema::Object(schema_object)
                    }
                }
            }
            .into())
        }
        Data::Enum(data_enum) => {
            let variant_schemas = generate_variant_schemas(&data_enum.variants)?;

            Ok(quote! {
                impl #impl_generics ::schemars::JsonSchema for #name #ty_generics #where_clause {
                    fn schema_name() -> String {
                        stringify!(#name).to_string()
                    }

                    fn json_schema(gen: &mut ::schemars::gen::SchemaGenerator) -> ::schemars::schema::Schema {
                        use ::schemars::schema::*;

                        let mut schema_object = SchemaObject {
                            instance_type: Some(InstanceType::String.into()),
                            ..Default::default()
                        };

                        let mut enum_values = Vec::new();
                        #variant_schemas

                        schema_object.enum_values = Some(enum_values);

                        Schema::Object(schema_object)
                    }
                }
            }
            .into())
        }
        Data::Union(_) => Err(syn::Error::new_spanned(
            name,
            "Schema derive macro does not support unions",
        )),
    }
}

/// Generate schema definitions for struct fields
fn generate_field_schemas(fields: &Fields) -> syn::Result<proc_macro2::TokenStream> {
    let mut field_tokens = Vec::new();

    match fields {
        Fields::Named(named_fields) => {
            for field in &named_fields.named {
                let field_name = field.ident.as_ref().unwrap();
                let field_name_str = field_name.to_string();
                let field_ty = &field.ty;

                // Extract description attribute
                let description = extract_description_attr(&field.attrs)?;

                let field_schema = if let Some(desc) = description {
                    quote! {
                        {
                            let mut field_schema = gen.subschema_for::<#field_ty>();
                            if let Schema::Object(ref mut obj) = field_schema {
                                obj.metadata = Some(Box::new(Metadata {
                                    description: Some(#desc.to_string()),
                                    ..Default::default()
                                }));
                            }
                            properties.insert(#field_name_str.to_string(), field_schema);
                            required.insert(#field_name_str.to_string());
                        }
                    }
                } else {
                    quote! {
                        {
                            let field_schema = gen.subschema_for::<#field_ty>();
                            properties.insert(#field_name_str.to_string(), field_schema);
                            required.insert(#field_name_str.to_string());
                        }
                    }
                };

                field_tokens.push(field_schema);
            }
        }
        Fields::Unnamed(_) => {
            return Err(syn::Error::new_spanned(
                fields,
                "Schema derive does not support tuple structs yet",
            ));
        }
        Fields::Unit => {
            // Unit structs have no fields
        }
    }

    Ok(quote! {
        #(#field_tokens)*
    })
}

/// Generate schema definitions for enum variants
fn generate_variant_schemas(
    variants: &Punctuated<Variant, Token![,]>,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut variant_tokens = Vec::new();

    for variant in variants {
        let variant_name = &variant.ident;
        let variant_str = variant_name.to_string();

        // For serde renamed values, we'd need to parse serde attributes
        // For now, use lowercase by default (common pattern)
        let variant_value = variant_str.to_lowercase();

        let _description = extract_description_attr(&variant.attrs)?;

        // Add the enum value
        variant_tokens.push(quote! {
            enum_values.push(::serde_json::json!(#variant_value));
        });
    }

    Ok(quote! {
        #(#variant_tokens)*
    })
}

/// Extract description from #[description = "..."] attribute
fn extract_description_attr(attrs: &[Attribute]) -> syn::Result<Option<String>> {
    for attr in attrs {
        if attr.path().is_ident("description") {
            if let Meta::NameValue(meta) = &attr.meta {
                if let syn::Expr::Lit(expr_lit) = &meta.value {
                    if let Lit::Str(lit) = &expr_lit.lit {
                        return Ok(Some(lit.value()));
                    }
                }
            }
        }
    }
    Ok(None)
}
