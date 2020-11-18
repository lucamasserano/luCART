-- Impurity functions, implemented as SQL functions.
-- Alex Reinhart

-- First, we make an aggregate function to calculate p, the fraction of entries
-- in a column that are 1.

-- Internal function; you should not need to use this.
CREATE FUNCTION fraction_ones_accum(cur_state integer[2], item integer)
RETURNS integer[] AS $$
        SELECT ARRAY[cur_state[1] + item, cur_state[2] + 1];
$$ LANGUAGE SQL;

-- Internal function; you should not need to use this.
CREATE FUNCTION fraction_ones_div(cur_state integer[2]) RETURNS double precision AS $$
       SELECT cur_state[1]::double precision / cur_state[2]::double precision;
$$ LANGUAGE SQL;

-- You can use fraction_ones like this:
-- SELECT fraction_ones(column_name) FROM table;
-- where column_name is a column of integers, either 0 or 1.
CREATE AGGREGATE fraction_ones (integer)
(
        sfunc = fraction_ones_accum,
        stype = integer[2],
        finalfunc = fraction_ones_div,
        initcond = '{0,0}' -- [number of 1s, number of entries]
);

-- Next, we implement the entropies using this function.
-- You can calculate entropy like this:
-- SELECT bayes_error(fraction_ones(column_name)) FROM table;
-- replacing bayes_error with any of the three functions below.

CREATE FUNCTION bayes_error(p double precision) RETURNS double precision AS $$
       SELECT LEAST(p, 1 - p);
$$ LANGUAGE SQL;

CREATE FUNCTION cross_entropy(p double precision) RETURNS double precision AS $$
       SELECT - p * ln(p) - (1 - p) * ln(1 - p);
$$ LANGUAGE SQL;

CREATE FUNCTION gini_index(p double precision) RETURNS double precision AS $$
       SELECT p * (1 - p);
$$ LANGUAGE SQL;
