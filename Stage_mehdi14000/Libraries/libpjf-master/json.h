/*
 * This file is part of libpjf
 * Copyright (C) 2011 Paweł Foremski <pawel@foremski.pl>
 * Copyright (C) 2009 ASN Sp. z o.o.
 * Copyright (C) 2008 Ivo van Poorten
 *
 * License: GNU Lesser General Public License version 2.1
 * Imported from zzjson library by: Pawel Foremski <pawel@foremski.pl>
 */

#ifndef _JSON_H_
#define _JSON_H_

typedef struct json {
	int depth;          /** recurrency depth */
	bool loose;         /** if true, be more permissive about standard strictness */

	const char *txt;    /** text representation */
	int i;              /** position in txt */
} json;

enum json_option {
	/** Accept a bit invalid syntax, which is easier to write by hand */
	JSON_LOOSE = 1
};

/** Create json parser */
json *json_create(void *mm);

/** Set parser options */
bool json_setopt(json *j, enum json_option o, long v);

/** Parse given string into unitype node
 * Never fails. In case of syntax error will return a unitype err object.
 * Given string is copied, not referenced */
ut *json_parse(json *j, const char *txt);

/** Print ut as text */
char *json_print(json *json, ut *var);

/** Helper function: escape string so it can be used in string representation */
char *json_escape(json *json, const char *str);

#endif
