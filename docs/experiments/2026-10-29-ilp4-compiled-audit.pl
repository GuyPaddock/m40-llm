#!/usr/bin/env perl
use strict;
use warnings;
use Digest::SHA qw(sha256_hex);
use JSON::PP;

sub read_bytes {
    open my $file, '<', $_[0] or die "$!";
    local $/;
    return <$file>;
}

my ($sass_path, $resource_path) = @ARGV;
die "provide SASS and resource paths\n" unless defined $resource_path;
my $reference_path = 'docs/experiments/2026-10-28-draft-compiled-audit.json';
my $reference = decode_json(read_bytes($reference_path));
my @expected = (@{$reference->{original_entries}}, $reference->{candidate});
my (%instructions, %assembly, $name);
for my $line (split /\n/, read_bytes($sass_path)) {
    $name = $1 if $line =~ /Function : (\S+)/;
    next unless defined $name;
    $assembly{$name} .= "$line\n";
    while ($line =~ /0x([a-f0-9]{16})/g) {
        push @{$instructions{$name}}, "0x$1\n";
    }
}
die 'kernel coverage differs' unless @expected == 138 && keys(%instructions) == 139;
for my $entry (@expected) {
    die "missing $entry->{name}" unless exists $instructions{$entry->{name}};
    my $words = $instructions{$entry->{name}};
    die "changed $entry->{name}" unless @$words == $entry->{words}
        && sha256_hex(join '', @$words) eq $entry->{sha256};
}
my @candidates = grep /gate_up_joint_ilp4/, keys %instructions;
die 'candidate coverage differs' unless @candidates == 1;
my $candidate = $candidates[0];
my $resources = read_bytes($resource_path);
$resources =~ /Function \Q$candidate\E:\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)/
    or die 'missing candidate resources';
my ($registers, $stack, $shared, $local) = ($1+0, $2+0, $3+0, $4+0);
die 'resource gate failed' unless $registers <= 48 && !$stack && !$local && !$shared;
my %chains;
while ($assembly{$candidate} =~ /FFMA (R\d+), R\d+(?:\.reuse)?, R\d+(?:\.reuse)?, \1\s*;/g) {
    $chains{$1}++;
}
die 'independent chains not found' unless keys(%chains) == 8;
my $words = $instructions{$candidate};
print JSON::PP->new->canonical->pretty->encode({
    is_passed => JSON::PP::true, unchanged_entries => 138,
    reference => $reference_path, sass_sha256 => sha256_hex(read_bytes($sass_path)),
    resource_sha256 => sha256_hex($resources),
    candidate => {name => $candidate, words => scalar(@$words),
        sha256 => sha256_hex(join '', @$words), registers => $registers,
        stack_bytes => $stack, local_bytes => $local, static_shared_bytes => $shared,
        dynamic_shared_bytes => 1536, accumulator_registers => [sort keys %chains]},
    scope => 'Compiled identity/resources, not measured occupancy, stalls or traffic'
});
