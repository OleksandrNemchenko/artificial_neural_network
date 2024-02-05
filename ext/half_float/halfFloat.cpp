
#include "halfFloat.hpp"


#define SINGLE_SIGN_SHIFT               (31)
#define SINGLE_EXP_SHIFT                (23)
#define SINGLE_MANT_SHIFT               (0)

#define SINGLE_SIGN_MASK                (0x80000000)
#define SINGLE_EXP_MASK                 (0x7F800000)
#define SINGLE_MANT_MASK                (0x007FFFFF)

#define SINGLE_POS_INFINITY             (0x7F800000)
#define SINGLE_NEG_INFINITY             (0xFF800000)

#define GET_SINGLE_SIGN_BIT(x)          ((x) >> SINGLE_SIGN_SHIFT)
#define GET_SINGLE_EXP_BITS(x)          (((x) >> SINGLE_EXP_SHIFT) & 0xFF)
#define GET_SINGLE_MANT_BITS(x)         ((x) & SINGLE_MANT_MASK)

#define SET_SINGLE_SIGN_BIT(x,dest)     ((dest) = ((((x) << SINGLE_SIGN_SHIFT) & SINGLE_SIGN_MASK) | ( (dest) & ( SINGLE_EXP_MASK  | SINGLE_MANT_MASK ))))
#define SET_SINGLE_EXP_BITS(x,dest)     ((dest) = ((((x) << SINGLE_EXP_SHIFT)  & SINGLE_EXP_MASK)  | ( (dest) & ( SINGLE_SIGN_MASK | SINGLE_MANT_MASK ))))
#define SET_SINGLE_MANT_BITS(x,dest)    ((dest) = ((((x) << SINGLE_MANT_SHIFT) & SINGLE_MANT_MASK) | ( (dest) & ( SINGLE_SIGN_MASK | SINGLE_EXP_MASK ))))

#define CONVERT_PATTERN( x )  ( reinterpret_cast<uint32_t *>( &x ) )

float16::float16() : m_uiFormat(0) {}

float16::float16( const float16 & rhs ) : m_uiFormat( rhs.m_uiFormat ) {}

float16::float16( const float & rhs )
{
    (*this) = rhs;
}

float16::~float16() {}

float16::operator float() const
{
    return ToFloat32( *this );
}

float16 & float16::operator = ( const float16 & rhs )
{
    m_uiFormat = rhs.m_uiFormat;

    return (*this);
}

float16 & float16::operator = ( const float & rhs )
{
    (*this) = ToFloat16( rhs );

    return (*this);
}

float16 & float16::operator = ( const double & rhs )
{
    (*this) = ToFloat16( static_cast<float>(rhs) );

    return (*this);
}

bool float16::operator == ( const float16 & rhs ) const
{
    return m_uiFormat == rhs.m_uiFormat;
}

bool float16::operator != ( const float16 & rhs ) const
{
    return !( (*this) == rhs );
}

float float16::ToFloat32( float16 rhs )
{
    float fOutput   = 0;                                  // floating point result
    uint32_t * uiOutput = CONVERT_PATTERN( fOutput );         // bit manipulated output

    if ( 0 == rhs.m_uiFormat )           return 0.0f;       // +zero
    else if ( 0x8000 == rhs.m_uiFormat ) return -0.0f;      // -zero

    uint32_t uiHalfSignBit   = GET_HALF_SIGN_BIT( rhs.m_uiFormat );
    uint32_t uiHalfMantBits  = GET_HALF_MANT_BITS( rhs.m_uiFormat ) << 13;
    int32_t  iHalfExpBits    = GET_HALF_EXP_BITS( rhs.m_uiFormat );

    //
    // Next we check for additional special cases:
    //

    if ( 0 == iHalfExpBits )
    {
        //
        // Denormalized values
        //

        SET_SINGLE_SIGN_BIT( uiHalfSignBit, (*uiOutput) );
        SET_SINGLE_EXP_BITS( 0, (*uiOutput) );
        SET_SINGLE_MANT_BITS( uiHalfMantBits, (*uiOutput) );
    }

    else if ( 0x1F == iHalfExpBits )
    {
        if ( 0 == uiHalfMantBits )
        {
            //
            // +- Infinity
            //

            (*uiOutput) = ( uiHalfSignBit ? SINGLE_NEG_INFINITY : SINGLE_POS_INFINITY );
        }
        else
        {
            //
            // (S/Q)NaN
            //

            SET_SINGLE_SIGN_BIT( uiHalfSignBit, (*uiOutput) );
            SET_SINGLE_EXP_BITS( 0xFF, (*uiOutput) );
            SET_SINGLE_MANT_BITS( uiHalfMantBits, (*uiOutput) );
        }
    }

    else
    {
        //
        // Normalized values
        //

        SET_SINGLE_SIGN_BIT( uiHalfSignBit, (*uiOutput) );
        SET_SINGLE_EXP_BITS( ( iHalfExpBits - 15 ) + 127, (*uiOutput) );
        SET_SINGLE_MANT_BITS( uiHalfMantBits, (*uiOutput) );
    }

    //
    // ATP: uiOutput equals the bit pattern of our floating point result.
    //

    return fOutput;
}

float16 float16::ToFloat16( float rhs )
{
    //
    // (!) Truncation will occur for values outside the representable range for float16.
    //   

    float16 fOutput;
    uint32_t uiInput  = *CONVERT_PATTERN( rhs );

    if ( 0.0f == rhs ) 
    { 
        fOutput.m_uiFormat = 0; 
        return fOutput;
    }
     
    else if ( -0.0f == rhs )
    {
        fOutput.m_uiFormat = 0x8000; 
        return fOutput;
    }

    uint32_t uiSignBit   = GET_SINGLE_SIGN_BIT( uiInput );
    uint32_t uiMantBits  = GET_SINGLE_MANT_BITS( uiInput ) >> 13;
     int32_t  iExpBits   = GET_SINGLE_EXP_BITS( uiInput );

    //
    // Next we check for additional special cases:
    //

    if ( 0 == iExpBits )
    {
        //
        // Denormalized values
        //

        SET_HALF_SIGN_BIT( uiSignBit, fOutput.m_uiFormat );
        SET_HALF_EXP_BITS( 0, fOutput.m_uiFormat );
        SET_HALF_MANT_BITS( uiMantBits, fOutput.m_uiFormat );
    }

    else if ( 0xFF == iExpBits )
    {
        if ( 0 == uiMantBits )
        {
            //
            // +- Infinity
            //

            fOutput.m_uiFormat = ( uiSignBit ? HALF_NEG_INFINITY : HALF_POS_INFINITY );
        }
        else
        {
            //
            // (S/Q)NaN
            //

            SET_HALF_SIGN_BIT( uiSignBit, fOutput.m_uiFormat );
            SET_HALF_EXP_BITS( 0x1F, fOutput.m_uiFormat );
            SET_HALF_MANT_BITS( uiMantBits, fOutput.m_uiFormat );
        }
    }

    else
    {
        //
        // Normalized values
        //

        int32_t iExponent = iExpBits - 127 + 15;

        if ( iExponent < 0 ) { iExponent = 0; }
        else if ( iExponent > 31 ) iExponent = 31;
            
        SET_HALF_SIGN_BIT( uiSignBit, fOutput.m_uiFormat );
        SET_HALF_EXP_BITS( iExponent, fOutput.m_uiFormat );
        SET_HALF_MANT_BITS( uiMantBits, fOutput.m_uiFormat );
    }

    //
    // ATP: uiOutput equals the bit pattern of our floating point result.
    //

    return fOutput;
}


float float16::ToFloat32Fast( float16 rhs )
{
    float fOutput   = 0;                                  // floating point result
    uint32_t * uiOutput = CONVERT_PATTERN( fOutput );         // bit manipulated output

    if ( 0 == rhs.m_uiFormat )           return 0.0f;       // +zero
    else if ( 0x8000 == rhs.m_uiFormat ) return -0.0f;      // -zero

    uint32_t uiHalfSignBit   = GET_HALF_SIGN_BIT( rhs.m_uiFormat );
    uint32_t uiHalfMantBits  = GET_HALF_MANT_BITS( rhs.m_uiFormat ) << 13;
     int32_t  iHalfExpBits   = GET_HALF_EXP_BITS( rhs.m_uiFormat );

    //
    // Normalized values
    //

    SET_SINGLE_SIGN_BIT( uiHalfSignBit, (*uiOutput) );
    SET_SINGLE_EXP_BITS( ( iHalfExpBits - 15 ) + 127, (*uiOutput) );
    SET_SINGLE_MANT_BITS( uiHalfMantBits, (*uiOutput) );

    //
    // ATP: uiOutput equals the bit pattern of our floating point result.
    //

    return fOutput;
}

float16 float16::ToFloat16Fast( float rhs )
{
    //
    // (!) Truncation will occur for values outside the representable range for float16.
    //   

    float16 fOutput;
    uint32_t uiInput  = *CONVERT_PATTERN( rhs );

    if ( 0.0f == rhs ) 
    { 
        fOutput.m_uiFormat = 0; 
        return fOutput;
    }
     
    else if ( -0.0f == rhs )
    {
        fOutput.m_uiFormat = 0x8000; 
        return fOutput;
    }

    uint32_t uiSignBit   = GET_SINGLE_SIGN_BIT( uiInput );
    uint32_t uiMantBits  = GET_SINGLE_MANT_BITS( uiInput ) >> 13;
     int32_t  iExpBits   = GET_SINGLE_EXP_BITS( uiInput );

    //
    // Normalized values
    //

    int32_t iExponent = iExpBits - 127 + 15;

    if ( iExponent < 0 ) { iExponent = 0; }
    else if ( iExponent > 31 ) iExponent = 31;
            
    SET_HALF_SIGN_BIT( uiSignBit, fOutput.m_uiFormat );
    SET_HALF_EXP_BITS( iExponent, fOutput.m_uiFormat );
    SET_HALF_MANT_BITS( uiMantBits, fOutput.m_uiFormat );

    //
    // ATP: uiOutput equals the bit pattern of our floating point result.
    //

    return fOutput;
}