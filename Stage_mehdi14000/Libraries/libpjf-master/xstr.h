/*
 * This file is part of libpjf
 * Copyright (C) 2011 Paweł Foremski <pawel@foremski.pl>
 * Copyright (C) 2005-2009 ASN Sp. z o.o.
 * Authors: Dawid Ciężarkiewicz <dawid.ciezarkiewicz@gmail.com> (original idea)
 *          Łukasz Zemczak <sil2100@asn.pl>
 *          Pawel Foremski <pawel@foremski.pl>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */

#ifndef __XSTR_H__
#define __XSTR_H__

#include <ctype.h>
#include <stdarg.h>

#include "mmatic.h"

typedef struct xstr {
	/** Pointer to allocated memory */
	char *s;

	/** Current length of *s */
	unsigned int len;

	/** Reserved memory; always a >= len + 1 */
	unsigned int a;
} xstr;

/** Create an xstr object
 * @param str    initial value; if NULL, "" is assumed
 * @return allocated memory */
xstr *xstr_create(const char *str, void *mm);
#define MMXSTR_CREATE(str) xstr_create((str), mm)

/** Init xstr, ie. set it to "" value */
void xstr_init(xstr *sx, void *mm);

/** Init xstr with value */
void xstr_init_val(xstr *sx, const char *ch, void *mm);

/** Return C string */
char *xstr_to_char(xstr *sx);

/** Duplicate string */
char *xstr_dup(xstr *sx, void *mm);

/** Allocate more space in buffer */
void xstr_reserve(xstr *sx, size_t l);

/** Append string to xstr */
void xstr_append(xstr *sx, const char *s);

/** Append string of length size to xstr */
void xstr_append_size(xstr *sx, const char *s, int size);

/** Append one char to xstr */
void xstr_append_char(xstr *sx, char s);

/** Set xstr value */
void xstr_set(xstr *xs, const char *s);

/** Set xstr value to string s of size size */
void xstr_set_size(xstr *xs, const char *s, int size);

/** Dellocate memory */
void xstr_free(xstr *xs);

/** Return the xstr as a standard char string */
#define xstr_string(xs) (xs ? xs->s : "")

/** Return the length of the string */
#define xstr_length(xs) (xs ? xs->len : 0)

/** Return (duplicated) stripped version of xs
 *
 * Caller must free allocated memory.
 * @return NULL on failure */
char *xstr_strip(xstr *xs);

/** Like xstr_strip() but on char* */
char *xstr_stripch(char *s, void *mm);

/** sprintf to xstr, allocating memory if necessary
 *
 * @param xs     xstr to be printed to (formatted)
 * @param format the printf-style format string
 * @return number of characters written */
int xstr_set_format(xstr *xs, const char *format, ...);

/** Append a sprintf-string to xstr, allocating memory if necessary
 *
 * @param xs     xstr to be printed to (formatted)
 * @param format the printf-style format string
 * @return number of characters appended */
int xstr_append_format(xstr *xs, const char *format, ...);

/** Cut the string @l byte starting from the end */
void xstr_cut(xstr *xs, unsigned int l);

#endif
