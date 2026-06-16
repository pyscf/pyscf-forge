#ifndef _INCLUDE_Properties_H
#define _INCLUDE_Properties_H
typedef struct struct_properties {
	char **keylist;
	char **values;
	int size;int N;
	const char *fpath;
} Properties;
Properties *new_Properties(const char *fpath,int size);
void Properties_update(Properties *this, const char *key,const char *value);
char *Properties_getvalue(Properties *this,const char *key);
void Properties_setvalue(Properties *this, const char *key,const char *type,const void *value);
int Properties_load(Properties *this, const char *fpath);
int Properties_save(Properties *this, const char *fpath);

#endif
