/*
 * tsort - topological sort
 *
 * This file is part of libpjf
 * Copyright (C) 2011 Paweł Foremski <pawel@foremski.pl>
 * Copyright (C) 2005-2009 ASN Sp. z o.o.
 * Copyright (C) 1998-2007 Free Software Foundation, Inc.
 *
 * Originally written by Mark Kettenis <kettenis@phys.uva.nl>.
 * Libified for libpjf by Pawel Foremski <pawel@foremski.pl>
 *
 * libpjf is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 *
 * libpjf is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "thash.h"
#include "tlist.h"
#include "mmatic.h"

/** A structure representing a dependency relation */
typedef struct _tsort_pair {
	/** What... */
	char *what;

	/** ...depends on this */
	char *dependson;
} tsort_pair;

/** Do a topological sort on input
 *
 * @param  input  a tlist of struct tsort_pair elements
 * @param  output an already initialized tlist to push() results to
 * @return 1 if successful
 * @return 0 otherwise
 */
int pjf_tsort(tlist *input, tlist *output, void *mm);
